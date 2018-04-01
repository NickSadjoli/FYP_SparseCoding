import numpy as np
from scipy.optimize import nnls
import unittest
import time
import copy

class Result(object):
    '''Result object for storing input and output data for omp.  When called from 
    `omp`, runtime parameters are passed as keyword arguments and stored in the 
    `params` dictionary.
    Attributes:
        Phi:  Predictor array after (optional) standardization.
        y:  Response array after (optional) standarization.
        ypred:  Predicted response.
        residual:  Residual vector.
        coef:  Solution coefficients.
        active:  Indices of the active (non-zero) coefficient set.
        err:  Relative error per iteration.
        params:  Dictionary of runtime parameters passed as keyword args.   
    '''
    
    def __init__(self, **kwargs):
        
        # to be computed
        self.Phi = None
        self.y = None
        self.ypred = None
        self.residual = None
        self.coef = None
        self.active = None
        self.err = None
        self.used_mp = None
        
        # runtime parameters
        self.params = {}
        for key, val in kwargs.iteritems():
            self.params[key] = val
            
    def update(self, coef, active, err, residual, ypred):
        '''Update the solution attributes.
        '''
        self.coef = coef
        self.active = active
        self.err = err
        self.residual = residual
        self.ypred = ypred



def norm2(v):
        return np.linalg.norm(v) / np.sqrt(len(v))

def l2_norm(x): #assume x is a vector matrix
    return np.sqrt(np.sum(np.abs(x)**2))

def check_rcvalue(rcov):
    i = np.argmax(rcov) #check the index location for maximum value in rcov, 
    rc = rcov[i]
    return rc, i


def mp_process(Phi, y, chosen_mp=None, ncoef=None, maxit=1000, tol=1e-3, ztol=1e-12, verbose=False):
    '''Main processing for creating sparse respresentation of an input y signal. Hosts several types
    of MP (Matching Pursuit) algorithms to choose from, depending on user input
    Args:
        Phi: Dictionary array of size m_samples * n_features. 
        y: Reponse array of size m_samples x 1.
        chosen_mp: indicating which MP algorithm to use for the process.
        cf: Extra argument, for use in BOMP or CoSaMP to either,
            - CoSaMP: Indicate how sparse should the resulting dictionary of CoSaMP should be
            - BOMP: Indicate how many slice of blocks should the signal y be sliced into.
        ncoef: Max number of coefficients(max sparsity).  Set to n_features/2 by default.
        tol: Convergence tolerance.  If relative error is less than
            tol * ||y||_2, exit.
        ztol: Residual covariance threshold.  If all coefficients are less 
            than ztol * ||y||_2, exit.
        verbose: Boolean, print some info at each iteration.
        
    Returns:
        result:  Result object.  See Result.__doc__
    '''
    # initialize result object
    result = Result(ncoef=ncoef, maxit=maxit,
                    tol=tol, ztol=ztol, used_mp=chosen_mp)
    '''
    if verbose:
        print(result.params)
    '''
    
    # check types, try to make somewhat user friendly
    if type(Phi) is not np.ndarray:
        Phi = np.array(Phi)
    if type(y) is not np.ndarray:
        y = np.array(y)
        
    # check that n_samples match
    if Phi.shape[0] != len(y):
        print('Phi and y must have same number of rows (samples)')
        return result
    
    # store arrays in result object    
    result.y = y
    result.Phi = Phi
    
    # for rest of call, want y to have ndim=1
    if np.ndim(y) > 1:
        y = np.reshape(y, (len(y),))

    # by default set max number of coef to half of total possible (as in half of # of column in Phi)
    if ncoef is None:
        ncoef = int(Phi.shape[1]/2)
    
    if chosen_mp == "omp":
        if verbose:
            print('\nIteration, relative error, number of non-zeros')
        return omp_loop(Phi, y, ncoef, maxit, tol, ztol, verbose, result)
    elif chosen_mp == "cosamp":
        if verbose:
            print('\nIteration, relative error, number of non-zeros')
        #return cosamp_loop(Phi, y, ncoef, maxit, tol, ztol, verbose, result)
        return cosamp(Phi, y, ncoef, epsilon=tol, max_iter=maxit, verbose=verbose)
    elif chosen_mp == "bomp":
        if verbose:
            print('\nIteration, relative error, number of non-zeros')
        return bomp_loop(Phi, y, ncoef, maxit, tol, ztol, verbose, result)
    elif chosen_mp == "bmpnils":
        '''
        if verbose:
            print('\nIteration, relative error, number of non-zeros')
        '''
        return bmpnils_loop(Phi, y, ncoef, maxit, tol, ztol, verbose, result)
    elif chosen_mp == "":
        return ()
    elif chosen_mp == None:
        print "No MP type chosen, please choose which MP you would like to choose!"
        return

    
    

def omp_loop(Phi, y, ncoef, maxit, tol, ztol, verbose, result_obj):
    '''
    Main loop for doing OMP representation of a y signal
    Inputs (Arguments):
        Phi: Dictionary array of size m_samples * n_features. 
        y: Reponse array (Input Signal, with optional noisy signal being used) of size m_samples x 1.
        ncoef: Max number of coefficients (max sparsity).  Set to n_features/2 by default.
        tol: Convergence tolerance or epsilon.  If relative error is less than
            tol * ||y||_2, exit.
        ztol: Residual covariance threshold.  If all coefficients are less 
            than ztol * ||y||_2, exit.
        verbose: Boolean, print some info at each iteration.
        result_obj: result object  for update and returned later at the end of this function loop
        
    Returns:
        result:  Result object.  See Result.__doc__
    '''
    # initialize things
    t0 = time.time() 
    result = result_obj
    Phi_transpose = Phi.T                        # store for repeated use
    active = []                                  # for storing the current active set used for iteration later
    coef = np.zeros(Phi.shape[1], dtype=float) # approximated solution vector
    cur_Phi = np.zeros(Phi.shape)
    residual = y                             # residual vector
    ypred = np.zeros(y.shape, dtype=float)   # predicted response from y minus residual
    ynorm = norm2(y)                         # store for computing relative err
    err = np.zeros(maxit, dtype=float)       # relative err vector, documenting error from result of every iteration

    # by default set max number of coef to half of total possible (as in half of # of column in Phi)
    if ncoef is None:
        ncoef = int(Phi.shape[1]/2)

    # Check if response has zero norm, because then we're done. This can happen
    # in the corner case where the response is constant and you normalize it.
    if ynorm < tol:     # the same as ||residual|| < tol * ||residual||
        print('Norm of the response is less than convergence tolerance.')
        result.update(coef, active, err[0], residual, ypred)
        return result
    
    # convert tolerances to relative
    tol = tol * ynorm       # convergence tolerance
    ztol = ztol * ynorm     # threshold for residual covariance
    # main loop
    for it in range(maxit):
        
        # compute residual covariance vector 
        rcov = np.dot(Phi_transpose, residual)

        #check threshold of covariance vector
        rc_val, i = check_rcvalue(rcov)
        if rc_val < ztol:
            if verbose:
                print('All residual covariances are below threshold.')
            break
        
        # update active set
        if i not in active:
            active.append(i)

        #print active
        cur_Phi[:,it] = Phi[:,i]
        cur_Phi_act = cur_Phi[:,0:it+1]
        # solve for new coefficients on new active set
        #print active
        #print np.shape(Phi[:, active])
        # coefi, _, _, _ = np.linalg.lstsq(Phi[:, active], y)
        coefi, _, _, _ = np.linalg.lstsq(cur_Phi_act, y)
        coef[active] = coefi   # update solution, i.e update the latest coefficient vector for current active set
        
        # update residual vector and error
        residual = y - np.dot(cur_Phi_act, coefi)
        ypred = y - residual
        err[it] = norm2(residual) / ynorm  
        
        # print status
        if verbose:
            print('{}, {}, {}'.format(it, err[it], len(active)))
            
        # check stopping criteria
        '''
        if err[it] < tol:  # converged
            if verbose:
                print('\nConverged.')
            break
        '''
        if l2_norm(residual) < tol:
            if verbose:
                print('\nConverged.')
            break
        
        
        if len(active) >= ncoef:   # hit max coefficients/sparsity
            if verbose:
                print('\nFound solution with max number of coefficients.')
            break
        

        if it == maxit-1:  # max iterations
            if verbose:
                print('\nHit max iterations.')
    
    result.update(coef, active, err[:(it+1)], residual, ypred)
    #return result
    tf = time.time()
    elapsed = round(tf - t0, 4)
    return coef, it, elapsed


def bomp_loop(Phi, y, s, maxit, tol, ztol, verbose, result_obj):
    '''
    Main loop for doing BOMP representation of a y signal
    Inputs (Arguments):
        Phi: Dictionary array of size m_samples * n_features. 
        y: Reponse array (Input Signal, with optional noisy signal being used) of size m_samples x 1.
        K: number of blocks to slice the y signal into.
        s: Max number of coefficients (max sparsity).  Set to n_features/2 by default.
        tol: Convergence tolerance or epsilon.  If relative error is less than
            tol * ||y||_2, exit.
        ztol: Residual covariance threshold.  If all coefficients are less 
            than ztol * ||y||_2, exit.
        verbose: Boolean, print some info at each iteration.
        result_obj: result object  for update and returned later at the end of this function loop
        
    Returns:
        result:  Result object.  See Result.__doc__
    '''

    def get_best_block(rcoef, Phi, s): # s = sparsity
        #rcoef is precalculated Phi_T with residual, thus having size (n x m) x (m x 1)
        
        #find armgax from abs of rcoef
        abs_rcoef = np.abs(rcoef) # find l2 norm along its vector (i.e. along columns)
        cols = np.shape(Phi)[1]

        u = np.argmax(abs_rcoef)
        rc_val = rcoef[u]
        # return the block index from Phi that contains the u *column* index, by using s. Note: Two different blocks can overlap with each other
        if u+((s+1)/2) >= cols:
            cur_set = range( u-((s-1)/2), cols) #create a current new set
            return u, cur_set, rcoef[u]
        elif u-((s-1)/2) < 0 :
            cur_set = range(0, u+((s+1)/2) + 1)
            return u, cur_set, rcoef[u]
        else:
            cur_set = range( u-((s-1)/2), u+((s+1)/2) + 1 ) #create a current new set
            return u, cur_set, rcoef[u]
        


    #print(Phi, y, s, maxit, tol, ztol, verbose)

    # initialize things
    t0 = time.time()
    result = result_obj
    Phi_transpose = Phi.T                                 # store for repeated use
    #Phi_block = np.array_split(Phi,K, axis=1)             # return splitting of (m x n) Phi -> K blocks of (m x j) matrices
    feature_len = np.shape(Phi)[1]                     # value of n from (m x n)
    active_blocks = None                                  # for storing the current concatenation of active blocks used for iteration
    active_set  = []                                    # storing list of chosen index values per iteration
    chosen_index = []
    residual = y                             # residual vector
    ypred = np.zeros(y.shape, dtype=float)   # predicted response from y minus residual
    ynorm = norm2(y)                         # store for computing relative err
    err = np.zeros(maxit, dtype=float)       # relative err vector, documenting error from result of every iteration
    final_loop = False

    # by default set max number of coef to half of total possible (as in half of # of column in Phi)
    if s is None:
        s = int(Phi.shape[1]/2)

    # Check if response has zero norm, because then we're done. This can happen
    # in the corner case where the response is constant and you normalize it.
    if ynorm < tol:     # the same as ||residual|| < tol * ||residual||
        if verbose:
            print('Norm of the response is less than convergence tolerance.')
            result.update(coef, active, err[0], residual, ypred)
        return coef, 0
    
    # convert tolerances to relative
    tol = tol * ynorm       # convergence tolerance
    ztol = ztol * ynorm     # threshold for residual covariance
    #print np.shape(Phi_transpose)

    # main loop
    for it in range(0,maxit): #note we're doing this until there are K blocks formed!
        
        # Calculate index for best suiting block, and check threshold of the resulting covariance vector for said block

        rcoef = np.dot(Phi_transpose, residual)
        max_index = np.argmax(np.abs(rcoef))
        rc_val = abs(rcoef[max_index])
    
        if rc_val < ztol:
            if verbose:
                print('All residual covariances are below threshold.')
            break
        
        #active_blocks = np.zeros(np.shape(Phi))
        if active_set is None: #create set if there isn't one already
            active_set = []
        if max_index not in chosen_index:
            chosen_index.append(max_index)
            # edge cases check for index values outside of range
            if (max_index+((s+1)/2) + 1) >= feature_len:
                cur_set = range( max_index-((s-1)/2), feature_len) #create a current new set
            elif max_index-((s-1)/2) < 0 :
                cur_set = range(0, max_index+((s+1)/2)+1)
            else:
                cur_set = range( max_index-((s-1)/2), max_index+((s+1)/2) + 1 ) #create a current new set
            
            for j in cur_set:
                if j not in active_set: #append this new set to the previously existing set
                    active_set.append(j)
        else:
            ''' 
            # edge cases check for index values outside of range
            if (max_index+((s+1)/2) + 1) >= feature_len:
                cur_set = range( max_index-((s-1)/2), feature_len) #create a current new set
            elif max_index-((s-1)/2) < 0 :
                cur_set = range(0, max_index+((s+1)/2) + 1)
            else:
                cur_set = range( max_index-((s-1)/2), max_index+((s+1)/2) + 1 ) #create a current new set
            for j in cur_set:
                if j not in active_set: #append this new set to the previously existing set
                    active_set.append(j)
            '''
            if norm2(residual) < tol:
                #print coef
                if verbose:
                    print ('\n Residual norm below epsilon value!')
                break
            else:
                if verbose:
                    print('\n BOMP has already chosen the same index somehow!')
                break
        

        active_blocks = np.zeros(Phi.shape)
        #print active_set, max_index
        active_blocks[:,active_set] = Phi[:,active_set]
        '''
        Update solution, Note that solving for least squares with new sized active blocks gives you exact same size of solution.
        This means that you don't need to relist or reconcatenate anything in this case (unlike normal omp)
        ''' 
        #coef, _, _, _ = np.linalg.lstsq(Phi[:, active_set], y)
        coef, _, _, _ = np.linalg.lstsq(active_blocks, y)
        '''
        m = np.zeros(rcoef.shape[0], dtype=float)
        m[active_set] = rcoef[active_set]
        '''
        # update residual vector and error
        #residual = y - np.dot(Phi[:,active_set], coef[active_set])
        residual = y - np.dot(active_blocks, coef)
        #residual = residual - np.dot(Phi, m)
        ypred = y - residual
        #err[it] = norm2(residual) / ynorm  
        
        # print status
        if verbose:
            print('{}, {}, {}'.format(it, err[it], len(active_set)))
            
        # check stopping criteria

        if (it == maxit):  # max blocks hit
            if verbose:
                print('\nHit max iterations.')
            break
        
        if norm2(residual) < tol:
            #print coef
            if verbose:
                print ('\n Residual norm below epsilon value!')
            break
        
        '''
        if final_loop:
            if verbose:
                print('\nMax amount of features reached')
            break
            '''
        if len(active_set) >= feature_len:
            if verbose:
                print('\nMax amount of features reached')
            break   
            
    
    #concatenate into one coefficient matrix
    result.update(coef, active_set, err[:(it+1)], residual, ypred)
    #return result
    tf = time.time()
    elapsed = round(tf-t0, 4)
    return coef, it, elapsed

def bmpnils_loop(Phi, y, s, maxit, tol, ztol, verbose, result_obj):
    '''
    Main loop for doing BOMP representation of a y signal
    Inputs (Arguments):
        Phi: Dictionary array of size m_samples * n_features. 
        y: Reponse array (Input Signal, with optional noisy signal being used) of size m_samples x 1.
        K: number of blocks to slice the y signal into.
        ncoef: Max number of coefficients (max sparsity).  Set to n_features/2 by default.
        tol: Convergence tolerance or epsilon.  If relative error is less than
            tol * ||y||_2, exit.
        ztol: Residual covariance threshold.  If all coefficients are less 
            than ztol * ||y||_2, exit.
        verbose: Boolean, print some info at each iteration.
        result_obj: result object  for update and returned later at the end of this function loop
        
    Returns:
        result:  Result object.  See Result.__doc__
    '''
    # initialize things
    t0 = time.time()
    result = result_obj
    feature_len = np.shape(Phi)[1]
    Phi_transpose = Phi.T                                 # store for repeated use
    block_indexes = []                                    # for listing currently active block indexes
    active_blocks = None                                  # for storing the current concatenation of active blocks used for iteration
    active_set = None                                    # storing list of chosen index values per iteration
    coef = None
    residual = y                             # residual vector
    ypred = np.zeros(y.shape, dtype=float)   # predicted response from y minus residual
    ynorm = norm2(y)                         # store for computing relative err
    err = np.zeros(maxit, dtype=float)       # relative err vector, documenting error from result of every iteration
    final_loop = False

    # by default set max number of coef to half of total possible (as in half of # of column in Phi)
    if s is None:
        s = int(Phi.shape[1]/2)

    # Check if response has zero norm, because then we're done. This can happen
    # in the corner case where the response is constant and you normalize it.
    if ynorm < tol:     # the same as ||residual|| < tol * ||residual||
        print('Norm of the response is less than convergence tolerance.')
        result.update(coef, active, err[0], residual, ypred)
        return result
    
    # convert tolerances to relative
    tol = tol * ynorm       # convergence tolerance
    ztol = ztol * ynorm     # threshold for residual covariance
    # main loop
    for it in range(0,maxit): #note we're doing this until there are K blocks!
        
        # Calculate index for best suiting block, and check threshold of the resulting covariance vector for said block
        rcoef = np.dot(Phi_transpose, residual)
        max_index = np.argmax(np.abs(rcoef))
        m = np.zeros(np.shape(Phi)[1])
        if active_set is None: #create set if there isn't one already
            active_set = []
        if max_index not in active_set:
            # edge cases check for index values outside of range
            if (max_index+((s+1)/2) +1) >= feature_len:
                cur_set = range( max_index-((s-1)/2), feature_len) #create a current new set
            elif max_index-((s-1)/2) < 0 :
                cur_set = range(0, max_index+((s+1)/2) + 1)
            else:
                cur_set = range( max_index-((s-1)/2), max_index+((s+1)/2) + 1 ) #create a current new set

            for j in cur_set:
                if j not in active_set: #append this new set to the previously existing set
                    active_set.append(j)
            m[active_set] = rcoef[active_set] #take the active indexes from calculated rcoef
        else:
            m[max_index] = rcoef[max_index] # no addition to the active set needed, since max_index is already part of it
        residual = residual - np.dot(Phi, m)
        
        # print status
        '''
        if verbose:
            #print('{}, {}, {}'.format(it, err[it], len(active_set)))
            print('It: {}'.format(it))
        '''
                    
        # check stopping criteria

        if (it == maxit):  # max iterations
            if verbose:
                print('\nHit max iterations.')
            break
        if len(active_set) >= feature_len:
            if verbose:
                print('\nAlready at max features!')
            break
        if norm2(residual) < tol:
            if verbose:
                print ('\n Residual norm below epsilon value!')
            break
    
    # use final active set to get result, using one least square algo
    shell = np.zeros(np.shape(Phi))
    shell[:, active_set] = Phi[:,active_set]
    result = np.linalg.lstsq(shell, y)
    tf = time.time()
    elapsed = round(tf-t0, 4)
    return result[0], it, elapsed


def cosamp(phi, y, s, epsilon=1e-10, max_iter=1000, verbose=False):
    """
    Return an `s`-sparse approximation of the target signal
    Input:
        - phi, sampling matrix
        - y, noisy sample vector
        - s, sparsity
    """
    t0 = time.time()
    a = np.zeros(phi.shape[1])
    residual = y
    it = 0 # count
    halt = False
    for it in range(1,max_iter):
        it += 1
        #if verbose:
        #    print("Iteration {}\r".format(it))
        
        P = np.dot(np.transpose(phi), residual)
        omega_set = np.argsort(P)[-(2*s):] # large components
        omega_set = np.union1d(omega_set, a.nonzero()[0]) # use set instead?
        phiOmega_set = phi[:, omega_set]
        b = np.zeros(phi.shape[1])

        # Solve Least Square for signal estimation. Note components picked in range of strongest Omega_set sets (Phi[:, Omega_set])
        b[omega_set], _, _, _ = np.linalg.lstsq(phiOmega_set, y)
        
        # Get new estimate
        b[np.argsort(b)[:-s]] = 0
        a = b
        
        # Halt criterion
        residual_old = residual
        residual = y - np.dot(phi, a)

        if it >= max_iter:
            if verbose:
                print("Hit max iterations!")
            break
        if np.linalg.norm(residual - residual_old) < epsilon:
            if verbose:
                print ("Converged to a certain value!")
            break
        if np.linalg.norm(residual) < epsilon:
            if verbose:
                print("Residual below epsilon!")
            break
        '''
        halt = it > max_iter or (np.linalg.norm(residual - residual_old) < epsilon) or np.linalg.norm(residual) < epsilon
         
         ''' 

        if verbose:
            print np.linalg.norm(residual - residual_old), np.linalg.norm(residual) 
        
            
    tf = time.time()
    elapsed = round(tf-t0, 4)
    if verbose:
        print("elapsed time: ", elapsed)
    return a, it, elapsed



def cosamp_loop(Phi, y, ncoef, maxit, tol, ztol, verbose, result_obj):
    '''Main loop to return a 's'-sparse representation of y using CoSaMP algorithm.
    Inputs (Arguments):
        Phi: Dictionary array of size n_samples * n_features. 
        x: Reponse array (Input Signal, with optional noisy signal being used) of size n_samples x 1.
        ncoef: Max number of coefficients/sparsity.  Set to n_features/2 by default.
        tol: Convergence tolerance or epsilon.  If relative error is less than
            tol * ||y||_2, exit.
        ztol: Residual covariance threshold.  If all coefficients are less 
            than ztol * ||y||_2, exit.
        verbose: Boolean, print some info at each iteration.
        result_obj: result object  for update and returned later at the end of this function loop
    Returns:
        result:  Result object.  See Result.__doc__
    '''

    # initialize things
    result = result_obj
    Phi_transpose = Phi.T                        # store for repeated use
    T_set = []                                   # strongest T support sets for each iteration
    approx = np.zeros(Phi.shape[1], dtype=float) # approximated solution vector
    residual = y                             # residual vector
    ypred = np.zeros(y.shape, dtype=float)   # predicted response from y minus residual
    ynorm = norm2(y)                         # store for computing relative err
    err = np.zeros(maxit, dtype=float)       # relative err vector, documenting error from result of every iteration 
    
    # Check if response has zero norm, because then we're done. This can happen
    # in the corner case where the response is constant and you normalize it.
    if ynorm < tol:     # the same as ||residual|| < tol * ||residual||
        print('Norm of the response is less than convergence tolerance.')
        result.update(coef, active, err[0], residual, ypred)
        return result
    
    # convert tolerances to relative
    tol = tol * ynorm       # convergence tolerance
    ztol = ztol * ynorm     # threshold for residual covariance
    
    if verbose:
        print('\nIteration, relative error, number of non-zeros')
   
    # main iteration
    for it in range(maxit):
        
        ''' 
        compute proxy signal (current residual covariance) based on current residual.
        Largest set in Proxy can be used to approximate largest set in x
        ''' 
        Proxy = np.dot(Phi_transpose, residual)

        #check proxy overall covariance value against ztol
        rc_val, _= check_rcvalue(Proxy)
        if rc_val < ztol:
            print('All residual covariances are below threshold.')
            break
        
        #take 2*ncoef set of largest components from the proxy. Note the sorting of Proxy first.
        omega = np.argsort(Proxy)[(-2*ncoef):]

        # Get current strongest active support set by combining omega with strongest support set from previous approx vector
        T_set = np.union1d(omega, approx.nonzero()[0])

        # prepare empty matrix same column size as Phi
        b = np.zeros(Phi.shape[1])

        
        '''
        Do signal estimation with least signal squares (lstsq).
        Note that we only Pick dictionary components from the calculated strongest T_set support sets (i.e. using Phi[:, T_set])
        '''
        b[T_set], _, _, _ = np.linalg.lstsq(Phi[:, T_set], y)

        '''
        Get new approximation by picking (Pruning) s largest sets (columns) from estimated signal
        HOW:sort from smallest to largest, and make every column from 0 to len-s (basically anything before the desired largest s columns)
        have 0 value. That way, you have a b with only last s columns containing the largest s sets as well
        '''
        b[np.argsort(b)[:-ncoef]] = 0

        #set obtained b as new approximation 
        approx = b 
        
        # update residual vector, predicted response (ypred) and error
        res_old = residual
        residual = y - np.dot(Phi, approx) #residual in next step
        ypred = y-residual
        res_dif = np.linalg.norm(residual-res_old) #check difference between current and previous residual
        err[it] = norm2(residual) / ynorm  #check for mean square root (norm2) between current residual and normalized x
        
        # print status
        if verbose and it == maxit-1:
            print('{}, {}, {}'.format(it, err[it], len(T_set)))
            
        if err[it] < tol:  # converged
            print('\nConverged.')
            break
        '''    
        if len(T_set) >= ncoef:   # hit max coefficients/sparsity
            print('\nFound solution with max number of coefficients.')
            break
        '''
        if it == maxit-1:  # max iterations
            print('\nHit max iterations.')
    
    result.update(approx, T_set, err[:(it+1)], residual, ypred)
    return result