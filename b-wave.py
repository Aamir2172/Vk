from fenics import *
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True

import sympy2fenics as sf
str2exp = lambda s: sf.sympy2exp(sf.str2sympy(s))


fileP = XDMFFile("outputs/convergence-Viscoelastic-time.xdmf")
fileP.parameters['rewrite_function_mesh']=False # for space convergence true
fileP.parameters["functions_share_mesh"] = True
fileP.parameters["flush_output"] = True

'''
Convergence test for time-discretisation of the wave equation with pure Neumann BCs
Newmark scheme

| rho u_tt - div(grad(u)) = f in Omega
|               grad(u).n = g on Gamma

'''

k      = 2
rho    = Constant(1.)
#alpha  = Constant(10.)


# ******* Exact solutions for error analysis ****** #

u_str     = 'exp(-t)*(x)*(y-1)'
#*(x-1)*(y-1))**2'
dt_u_str  = '(-exp(-t))*(x)*(y-1)'
#*(x-1)*(y-1))**2'0
dtt_u_str = '(exp(-t))*(x)*(y-1)'
#*(x-1)*(y-1))**2'

mesh = UnitSquareMesh(64,64)
n = FacetNormal(mesh); he = FacetArea(mesh) 
vol = CellVolume (mesh)
#k1=2.0
a = 4.0
sigma = 3.0 * a * k * (k - 1) / 8.0 * he("+") **2 * avg (1 / vol )
sigma_boundary =3.0 * a * k * (k - 1) * he **2 / vol
# ********* Finite dimensional spaces ********* #

#Hh = FunctionSpace(mesh, 'CG', k)
P2 = FiniteElement("CG", mesh.ufl_cell(),2)
Hh = FunctionSpace(mesh,P2)#MixedElement([P2,P2]))

print('dofs = ', Hh.dim())

print('h = ',mesh.hmax())

# ********* test and trial functions for product space ****** #
u = TrialFunction(Hh)
v = TestFunction(Hh)

Tfinal = 2; 
dtvec = [1./2,1./4, 1./8, 1./16, 1./32, 1./64] 
nkmax = 3

es_0 = []; rs_0 = []; es_div = []; rs_div = []
es = []; rs = []; es_jump = []; rs_jump = []

rs.append(0); rs_0.append(0); rs_div.append(0); rs_jump.append(0); 

# ***** Error analysis ***** #

for nk in range(nkmax):
    dt = dtvec[nk]
    print("....... Refinement level : dt = ", dt)
    
    # ********* instantiation of initial conditions ****** #
    t =0.
    
    u_ex = Expression(str2exp(u_str), t = t, degree=k+4, domain=mesh)
    u_exn = Expression(str2exp(u_str), t = 0, degree=k+4, domain=mesh)
    u_exnp1 = Expression(str2exp(u_str), t = 0, degree=k+4, domain=mesh)
    u_exnm1 = Expression(str2exp(u_str), t = 0, degree=k+4, domain=mesh)
    u_exM = Expression(str2exp(u_str), t = t-dt, degree=k+4, domain=mesh)
#    dt_u_ex = Expression(str2exp(dt_u_str), t = t, degree=k+4, domain=mesh)
#    dtt_u_ex = Expression(str2exp(dtt_u_str), t = t, degree=k+4, domain=mesh)
    dtt_u_exn = Expression(str2exp(dtt_u_str), t = 0, degree=k+4, domain=mesh)
    dtt_u_exnp1 = Expression(str2exp(dtt_u_str), t = 0, degree=k+4, domain=mesh)
    dtt_u_exnm1 = Expression(str2exp(dtt_u_str), t =0, degree=k+4, domain=mesh)

    u_old = interpolate(u_ex,Hh)
    u_oold = interpolate(u_exM,Hh)
        
    # initialise L^infty(0,T;H) norm (max over the time steps)
    E_s0 = 0; E_sdiv = 0; E_sjump=0; E_s = 0
    
    # ********* Time loop ************* # 
    while (t < Tfinal):

        if t + dt > Tfinal:
            dt = Tfinal - t
            t  = Tfinal
        else:
            t += dt
            
        print("t=%.6f" % t)
        
        def hess(w):
            return grad(grad(w))

        def a_h(w1,w2):
            bilinear = inner(hess(w1),hess(w2))*dx \
                     - inner(avg(dot(hess(w1)*n,n)),jump(grad (w2),n))*dS \
                     - inner(jump(grad(w1),n),avg(dot(hess(w2)*n,n)))*dS \
                     + sigma/he("+")*inner(jump(grad(w1),n),jump(grad(w2),n))*dS\
                     - inner(dot(hess(w1)*n,n),dot(grad(w2),n))*ds\
                     - inner(dot(grad(w1),n),dot(hess(w2)*n,n))*ds \
                     + sigma_boundary/he*inner(dot(grad(w1),n),dot(grad(w2),n))*ds
            return bilinear

        def energynorm(w,w3):
            norm = pow(assemble((((grad(grad(w)))**2)*dx)+(sigma/(he("+")))*((jump(grad(w),n))**2)*dS+((sigma_boundary/(he))*((dot(grad(w3),n))**2)*ds)),0.5)
            return norm
        
        u_exnp1.t = t;
        u_exn.t = t-dt;
        u_exnm1.t = t-2*dt;
        
        dtt_u_exnp1.t = t;
        dtt_u_exn.t = t-dt;
        dtt_u_exnm1.t = t-2*dt;

        u_ex_mid  = 0.5*(u_exnp1+u_exn)
        dt_uex=1./dt*(u_exnp1-u_exn)
        
        u_exhat =0.25*(u_exnp1+2*u_exn+u_exnm1)
        u_hat = 0.25*(u+2*u_old+u_oold)
        fhat_ex = 0.25*(rho*dtt_u_exnp1 + div(grad(div(grad(u_exnp1)))) \
                        + 2*rho*dtt_u_exn + 2*div(grad(div(grad(u_exn)))) \
                        + rho*dtt_u_exnm1 + div(grad(div(grad(u_exnm1)))))
       
        FF =  (1/dt**2)*(u-2*u_old+u_oold)*v*dx \
            + 1/rho*a_h(u_hat,v)\
            - 1/rho*fhat_ex * v * dx\
            + 1/rho*inner(dot(grad(u_exhat),n),dot(hess(v)*n,n))*ds\
            - 1/rho*sigma_boundary/he*inner(dot(grad(u_exhat),n),dot(grad(v),n))*ds
         
        AA,BB = system(FF)
        u_h = Function(Hh)
        solve(AA==BB, u_h, DirichletBC(Hh,u_exnp1,"on_boundary"))
        
        u_mid = 0.5*(u_h+u_old)
        dt_u= 1./dt*(u_h-u_old)
        
        #if t< Tfinal:
        # compute L^infty(0,T;H) errors of each contribution at time t_{n+1/2} 
        
        E_s0 = max(E_s0,pow(assemble((dt_uex-dt_u)**2*dx),0.5))
        E_sdiv= max(E_sdiv,energynorm(u_ex_mid-u_mid,u_ex_mid-u_mid,))
               
        assign(u_oold,u_old)
        assign(u_old,u_h)

    
        
    # ********* Storing errors ****** #
    
    es_0.append(E_s0)
    es_div.append(E_sdiv)
   
    
    if(nk>0):
        rs_0.append(ln(es_0[nk]/es_0[nk-1])/ln(dtvec[nk]/dtvec[nk-1]))
        rs_div.append(ln(es_div[nk]/es_div[nk-1])/ln(dtvec[nk]/dtvec[nk-1]))
 # ********* Generating error history ****** #
print('==================================================================================================================')
print('  dt    &    L2    &   Rate   &   Energy  &  Rate ')
print('==================================================================================================================')

print('=====================================================================================')
for nk in range(nkmax):
    print('{:.6f}  & {:6.2e} & {:.3f} & {:6.2e} & {:.3f} '.format(dtvec[nk], es_0[nk], rs_0[nk], es_div[nk], rs_div[nk]))
print('=============================================================================================')


