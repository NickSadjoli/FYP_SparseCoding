import numpy as np
from scipy.optimize import nnls
import unittest
import time
import copy

def norm2(v):
        return np.linalg.norm(v) / np.sqrt(len(v))

def l2_norm(x): #assume x is a vector matrix
    return np.sqrt(np.sum(np.abs(x)**2))

def check_rcvalue(rcov):
    i = np.argmax(rcov) #check the index location for maximum value in rcov, 
    rc = rcov[i]
    return rc, i

def omp(Phi, y, ncoef=None, maxit=1000, tol=1e-3, ztol=1e-12, verbose=False):
    '''
     Function for performing OMP for sparse representation of input signal y

     Inputs:
     Phi => Sparse Dictionary to be used
     y   => signal to be sparse represented
     ncoef => sparsity of resulting output signal
     tol => convergance tolerance value (epsilon). If relative error < tol * ||y||_2, exit
     ztol => residual covariance threshold value
     verbose => Whether or not to display information during run.
    '''

    # initialize things
    #y = np.reshape(y, (len(y), 1))
    t0 = time.time()                           # start recording of runtime for entire loop
    Phi_transpose = Phi.T                        # store for repeated use
    active = []                                  # for storing the current active set used for iteration later
    coef = np.zeros(Phi.shape[1], dtype=float) # approximated solution vector
    #coef = None
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
        return coef, 0, 0
    
    # convert tolerances to relative
    tol = tol * ynorm       # convergence tolerance
    ztol = ztol * ynorm     # threshold for residual covariance
    # main loop
    for it in range(maxit):
        
        # compute residual covariance vector 
        rcov = np.dot(Phi_transpose, residual)

        # pick index of maximum value in rcov
        i = np.argmax(rcov)

        #check threshold of covariance vector
        rc_val = rcov[i]

        if rc_val < ztol:
            if verbose:
                print('All residual covariances are below threshold.')
                return coef, 0, 0
        
        # update active set
        if i not in active:
            active.append(i)
        
        # else if i is already inside active set
        else: 
            if verbose:
                print(i, "Repeating i detected!")
            res = np.delete(rcov, i)
            #np.argsort(rcov, kind='heapsort')
            j= np.argmax(res)
            if j not in active: 
                active.append(j)
            else:
                if verbose:
                    print(j, "2nd largest still in active, break!")
                break
        
        if verbose:
            print active

        cur_Phi[:, it] = Phi[:,i] #record the current chosen Phi column in the current iteration
        
        '''#version using customized cur_phi_act
        cur_Phi_act = cur_Phi[:,0:it+1]
        coefi, _, _, _ = np.linalg.lstsq(cur_Phi_act, y)
        if verbose:
            print np.shape(coefi)
        coef[active] = np.reshape(coefi,np.shape(coef[active]))
        '''

        #version using only normal curPhi from above, no bs
        coefi, _, _, _ = np.linalg.lstsq(cur_Phi[:,0:it+1], y)
        if verbose:
            print np.shape(coefi)
        coef[active] = np.reshape(coefi,np.shape(coef[active]))

        ''' #version with only using active array
        print np.shape(Phi[:, active])
        cur_Phi[:,active] = Phi[:,active]
        coefi, _, _, _ = np.linalg.lstsq(Phi[:, active], y)
        '''
        
        

           # update solution, i.e update the latest coefficient vector for current active set
        
        # update residual vector and error

        #version using only normal cur_phi
        residual = y - np.dot(cur_Phi[:,0:it+1], coefi)

        #version using customized cur_phi_act
        #residual = y - np.dot(cur_Phi_act, coefi)

        #version with only using active array
        #residual = y - np.dot(Phi[:, active].T, y)

        ypred = y - residual
        err[it] = norm2(residual) / ynorm  
        
        # print status
        if verbose:
            print('{}, {}, {}, {}'.format(it, err[it], len(active), l2_norm(residual)))
            
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

    tf = time.time()
    elapsed = round(tf - t0, 4)
    return coef, it, elapsed



def bomp(Phi, y, ncoef=None, maxit=1000, tol=1e-3, ztol=1e-12, verbose=False):
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
        if u+((ncoef+1)/2) >= cols:
            cur_set = range( u-((ncoef-1)/2), cols) #create a current new set
            return u, cur_set, rcoef[u]
        elif u-((ncoef-1)/2) < 0 :
            cur_set = range(0, u+((ncoef+1)/2) + 1)
            return u, cur_set, rcoef[u]
        else:
            cur_set = range( u-((ncoef-1)/2), u+((ncoef+1)/2) + 1 ) #create a current new set
            return u, cur_set, rcoef[u]
        


    #print(Phi, y, s, maxit, tol, ztol, verbose)

    # initialize things
    t0 = time.time()
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
    if ncoef is None:
        ncoef = int(Phi.shape[1]/2)

    # Check if response has zero norm, because then we're done. This can happen
    # in the corner case where the response is constant and you normalize it.
    if ynorm < tol:     # the same as ||residual|| < tol * ||residual||
        if verbose:
            print('Norm of the response is less than convergence tolerance.')
            return coef, 0, 0
    
    # convert tolerances to relative
    tol = tol * ynorm       # convergence tolerance
    ztol = ztol * ynorm     # threshold for residual covariance
    #print np.shape(Phi_transpose)

    # main loop
    for it in range(0,maxit): #note we're doing this until there are K blocks formed!
        
        # Calculate index for best suiting block, and check threshold of the resulting covariance vector for said block

        rcoef = np.dot(Phi_transpose, residual)
        max_index = np.argmax(np.abs(rcoef))
        if verbose:
                print('max_index = ', max_index)
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
            if (max_index+((ncoef+1)/2) + 1) >= feature_len:
                cur_set = range( max_index-((ncoef-1)/2), feature_len) #create a current new set
            elif max_index-((ncoef-1)/2) < 0 :
                cur_set = range(0, max_index+((ncoef+1)/2)+1)
            else:
                cur_set = range( max_index-((ncoef-1)/2), max_index+((ncoef+1)/2) + 1 ) #create a current new set
            
            for j in cur_set:
                if j not in active_set: #append this new set to the previously existing set
                    active_set.append(j)
        else:
            ''' 
            # edge cases check for index values outside of range
            if (max_index+((ncoef+1)/2) + 1) >= feature_len:
                cur_set = range( max_index-((ncoef-1)/2), feature_len) #create a current new set
            elif max_index-((ncoef-1)/2) < 0 :
                cur_set = range(0, max_index+((ncoef+1)/2) + 1)
            else:
                cur_set = range( max_index-((ncoef-1)/2), max_index+((ncoef+1)/2) + 1 ) #create a current new set
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
        if verbose:
                print('calculating coef')
        '''
        Update solution, Note that solving for least squares with new sized active blocks gives you exact same size of solution.
        This means that you don't need to relist or reconcatenate anything in this case (unlike normal omp)
        ''' 
        #coef, _, _, _ = np.linalg.lstsq(Phi[:, active_set], y)
        coef, _, _, _ = np.linalg.lstsq(active_blocks, y)
        if verbose:
                print('calculated coef.')
        '''
        m = np.zeros(rcoef.shape[0], dtype=float)

        m[active_set] = rcoef[active_set]
        '''
        # update residual vector and error
        #residual = y - np.dot(Phi[:,active_set], coef[active_set])
        residual = y - np.dot(active_blocks, coef)
        if verbose:
                print('calculated residual.')
        #residual = residual - np.dot(Phi, m)
        ypred = y - residual
        #err[it] = norm2(residual) / ynorm  
        
        # print status
        if verbose:
            print('{}, {}, {}'.format(it, err[it], len(active_set)))
        
        #print('{}, {}, {}'.format(it, err[it], len(active_set)))

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
    tf = time.time()
    elapsed = round(tf-t0, 4)
    return coef, it, elapsed



def bmpnils(Phi, y, ncoef=None, maxit=1000, tol=1e-3, ztol=1e-12, verbose=False):
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
        
    Returns:
        result:  Result object.  See Result.__doc__
    '''
    # initialize things
    t0 = time.time()
    feature_len = np.shape(Phi)[1]
    Phi_transpose = Phi.T                                 # store for repeated use
    block_indexes = []                                    # for listing currently active block indexes
    active_blocks = None                                  # for storing the current concatenation of active blocks used for iteration
    active_set = None                                    # storing list of chosen index values per iteration
    coef = None
    residual = np.reshape(y, (len(y), ))                 # residual vector, first resizing to ensure consistent calculations later on
    ypred = np.zeros(y.shape, dtype=float)   # predicted response from y minus residual
    ynorm = norm2(y)                         # store for computing relative err
    err = np.zeros(maxit, dtype=float)       # relative err vector, documenting error from result of every iteration
    final_loop = False
    #m = np.zeros((np.shape(Phi)[1],1))

    # by default set max number of coef to half of total possible (as in half of # of column in Phi)
    if ncoef is None:
        ncoef = int(Phi.shape[1]/2)

    # Check if response has zero norm, because then we're done. This can happen
    # in the corner case where the response is constant and you normalize it.
    if ynorm < tol:     # the same as ||residual|| < tol * ||residual||
        if verbose:
            print('Norm of the response is less than convergence tolerance.')
        #result.update(coef, active, err[0], residual, ypred)
        return coef, 0, 0
    
    # convert tolerances to relative
    tol = tol * ynorm       # convergence tolerance
    ztol = ztol * ynorm     # threshold for residual covariance
    # main loop
    for it in range(0,maxit): #note we're doing this until there are K blocks!
        
        # Calculate index for best suiting block, and check threshold of the resulting covariance vector for said blocks
        rcoef = np.dot(Phi_transpose, residual)
        #print np.shape(rcoef)
        max_index = np.argmax(np.abs(rcoef))
        m = np.zeros(np.shape(Phi)[1])
        #m.fill(0.0) # zeroes all values in m
        #print np.shape(m)
        if active_set is None: #create set if there isn't one already
            active_set = []
        if max_index not in active_set: 
            # edge cases check for index values outside of range
            if (max_index+((ncoef+1)/2) +1) >= feature_len:
                cur_set = range( max_index-((ncoef-1)/2), feature_len) #create a current new set
            elif max_index-((ncoef-1)/2) < 0 :
                cur_set = range(0, max_index+((ncoef+1)/2) + 1)
            else:
                cur_set = range( max_index-((ncoef-1)/2), max_index+((ncoef+1)/2) + 1 ) #create a current new set

            for j in cur_set:
                if j not in active_set: #append this new set to the previously existing set
                    active_set.append(j)
            cur_rcoef = rcoef[active_set]
            m[active_set] = rcoef[active_set] #take the appendctive indexes from calculated cur_rcoef
            it = it + (ncoef-1) #set it = it + ncoef for the next iteration

        else:       
            m[max_index] = rcoef[max_index] # no addition to the active set needed, since max_index is already part of it

        residual = residual - np.dot(Phi, m)
        
        # print status

        if verbose:
            #print('{}, {}, {}'.format(it, err[it], len(active_set)))
            print('It: {}'.format(it))
                    
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




def cosamp(phi, y, ncoef=None, maxit=1000, tol=1e-3, ztol=1e-12, verbose=False):
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


    s = ncoef
    epsilon = tol
    max_iter = maxit

    for it in range(1,max_iter):
        it += 1
        #if verbose:
        #    print("Iteration {}\r".format(it))
        
        P = np.dot(np.transpose(phi), residual)
        omega_set = np.argsort(P, axis=0, kind='heapsort')[-(2*s):] # large components, result size = [n, 1]
        #omega_set = np.reshape(omega_set, (len(omega_set), ) )
        current_nonzero = a.nonzero()[0] # result size = [n, ], need to reshape for union1d
        current_nonzero = np.reshape(current_nonzero, (len(current_nonzero), 1) )
        #omega_set = np.union1d(omega_set, a.nonzero()[0]) # use set instead?
        omega_set = np.union1d(omega_set, current_nonzero)
        phiOmega_set = phi[:, omega_set]
        b = np.zeros(phi.shape[1])
        
        # Solve Least Square for signal estimation. Note components picked in range of strongest Omega_set sets (Phi[:, Omega_set])
        #b[omega_set], _, _, _ = np.linalg.lstsq(phiOmega_set, y)
        coef, _, _, _ = np.linalg.lstsq(phiOmega_set, y)
        b[omega_set] = np.reshape(coef, (len(coef), ))
        
        # Get new estimate
        b[np.argsort(b)[:-s]] = 0
        a = b
        
        # Halt criterion
        residual_old = residual
        #print np.shape(y), np.shape(np.dot(phi, a))
        coef_vect = np.dot(phi, a)
        coef_vect = np.reshape(coef_vect, (len(coef_vect), 1) )
        #residual = y - np.dot(phi, a)
        residual = y - coef_vect
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

