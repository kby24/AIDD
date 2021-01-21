import torch
from scipy.optimize import linear_sum_assignment
def sgraphmatch(A,B,m,iteration):
    totv = A.shape[0]
    n = totv-m
    start = torch.ones(n,n).cuda()*(1/n)

    if m!= 0:

        A12 = A[:m,m:totv]
        A21 = A[m:totv,:m]
        B12 = B[:m,m:totv]
        B21 = B[m:totv,:m]
    
    if m == 0:
        A12 = A21 = B12 = B21 = torch.zeros_like(n,n)
    
    if n==1:
        A12 = A12.T
        A21 = A21.T
        B12 = B12.T
        B21 = B21.T


    A22 = A[m:totv,m:totv]
    B22 = B[m:totv,m:totv]

    tol = 1

    patience = iteration
    P = start

    toggle = 1
    iter = 0


    x = torch.mm(A21,B21.T)
    y = torch.mm(A12.T,B12)

    while (toggle == 1 and iter < patience):

        iter = iter + 1

        z = torch.mm(torch.mm(A22,P),B22.T)
        w = torch.mm(torch.mm(A22.T,P),B22)
        Grad = x + y + z + w
        
        mm = abs(Grad).max() 
        
        
        obj = Grad+torch.ones([n,n]).cuda()*mm

        _,ind = linear_sum_assignment(-obj.cpu())

        Tt = torch.eye(n).cuda()
        Tt = Tt[ind]
        
        wt = torch.mm(torch.mm(A22.T,Tt),B22)
        
        

        c = torch.sum(torch.diag(torch.mm(w,P.T)))
        
           
        d = torch.sum(torch.diag(torch.mm(wt,P.T)))+torch.sum(torch.diag(torch.mm(wt,Tt.T))) 
        e = torch.sum(torch.diag(torch.mm(wt,Tt.T)))

        
        u = torch.sum(torch.diag(torch.mm(P.T,x) + torch.mm(P.T,y)))
        v = torch.sum(torch.diag(torch.mm(Tt.T,x)+torch.mm(Tt.T,y)))
            
                
        if (c - d + e == 0 and d - 2 * e + u - v == 0):
            alpha = 0
        else: 
            alpha = -(d - 2 * e + u - v)/(2 * (c - d + e))


        f0 = 0
        f1 = c - e + u - v

        falpha = (c - d + e) * alpha**2 + (d - 2 * e + u - v) * alpha

        if (alpha < tol and alpha > 0 and falpha > f0 and falpha > f1):

            P = alpha * P + (1 - alpha) * Tt

        elif f0 > f1:
            P = Tt
            
        else: 
            P = Tt
            toggle = 0
        break
    

    D = P
    _,corr = linear_sum_assignment(-P.cpu()) # matrix(solve_LSAP(P, maximum = TRUE))# return matrix P 

    
    corr = torch.LongTensor(corr).cuda()
    P = torch.eye(n).cuda()
    
    

    ccat = torch.cat([torch.eye(m).cuda(),torch.zeros([m,n]).cuda()],1)
    P = torch.index_select(P,0,corr)
    
   
    rcat = torch.cat([torch.zeros([n,m]).cuda(),P],1)

    
    P =  torch.cat((ccat,rcat),0)
    # P =  np.vstack([np.hstack([torch.eye(m),torch.zeros([m,n])]),np.hstack([np.zeros([n,m]),P[corr]])]) 
    corr = corr 
    
    return corr,P
