#C0IP for von Karman equation
from dolfin import *
import sympy2fenics as sf

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
# refinement steps

# additional functions
str2exp = lambda s: sf.sympy2exp(sf.str2sympy(s))
# ***** Model coefficients and parameters ***** #

u_str='exp(-t)*(x*(1-x)*y*(1-y))**2' 
dt_u_str='(-exp(-t))*(x*(1-x)*y*(1-y))**2'
dtt_u_str='exp(-t)*(x*(1-x)*y*(1-y))**2'
v_str='cos(t)*(sin(pi*x)*sin(pi*y))**2'
#'t**3*(x*(1-x)*y*(1-y))**2'
#((t))*(sin(pi*x)*sin(pi*y))**2'


hh = [];
nn = [];
du = [];
es_0=[];
es_H2=[];
rs_0=[];
rs_H2=[];
rs_0.append(0.0);
rs_H2.append(0.0);
esv_0=[];
esv_H2=[];
rsv_0=[];
rsv_H2=[];
rsv_0.append(0.0);
rsv_H2.append(0.0);
Tfinal=1
mesh = UnitSquareMesh(32,32)
n = FacetNormal(mesh);
hh.append(mesh.hmax())
he = FacetArea(mesh);

vol = CellVolume (mesh)
k=2.0
a = 4.0
sigma = 3.0 * a * k * (k - 1) / 8.0 * he("+") **2 * avg (1 / vol )
sigma_boundary =3.0 * a * k * (k - 1) * he **2 / vol

P2 = FiniteElement("CG", mesh.ufl_cell(),2)
Hh = FunctionSpace(mesh,MixedElement([P2,P2]))
nn.append(Hh.dim())
def VK(w1, w2):
    return (w1.dx(0).dx(0))*(w2.dx(1).dx(1)) + (w1.dx(1).dx(1))*(w2.dx(0).dx(0)) - 2.0 *(w1.dx(0).dx(1))*(w2.dx(0).dx(1))
    
# ***** Error analysis ***** #
def hess(w):
    return grad(grad(w))
def stiffness(w1,w2):
    bilinear = inner(hess(w1),hess(w2)) * dx \
             - inner(avg(dot(hess(w1)*n,n)),jump(grad (w2),n))*dS \
             - inner(dot(grad(w1),n),dot(hess(w2)*n,n))*ds \
             - inner(jump(grad(w1),n),avg(dot(hess(w2)*n,n)))*dS \
             - inner(dot(hess(w1)*n,n),dot(grad(w2),n))*ds \
             + sigma/he("+")*inner(jump(grad(w1),n),jump(grad(w2),n))*dS \
             + sigma_boundary/he*inner(dot(grad(w1),n),dot(grad(w2),n))*ds
    return bilinear
def energynorm(w):
    norm=pow(assemble((((grad(grad(w)))**2)*dx)+(sigma/(he("+")))*((jump(grad(w),n))**2)*dS+((sigma_boundary/(he))*((dot(grad(w),n))**2)*ds)),0.5)
    return norm
#save data    
file_vk = XDMFFile(mesh.mpi_comm(), "vk.xdmf")
file_vk.parameters['rewrite_function_mesh']=False
file_vk.parameters["functions_share_mesh"] = True
file_vk.parameters["flush_output"] = True
 


dtvec = [1./4,1./8,1./16,1./32,1./64]#[1./32,1./64,1./128]#,1./50,1./45]#,1./16]#$,1./12,1./14]#,1./16]#,1./64]#,1./128,1.256]#,1./64]#,1./128] 
nkmax =3

for nk in range(nkmax):
    dt=dtvec[nk]
    print("....... Refinement level : nk = ", nk)
    print("....... DoF = ", Hh.dim())
    t=dt
    
    u_ex0 = Expression(str2exp(u_str), domain=mesh,t=t-dt,dt=dt,degree=6) 
    u_ex1 = Expression(str2exp(u_str), domain=mesh,t=t,degree=6)
    
    v_ex0 =Expression(str2exp(v_str), domain=mesh,t=t-dt,dt=dt,degree=6) 
    v_ex1 = Expression(str2exp(v_str), domain=mesh,t=t,degree=6)
    
    dtt_u_ex0 = Expression(str2exp(dtt_u_str), domain=mesh,t=t-dt,dt=dt,degree=6)
    dtt_u_ex1 = Expression(str2exp(dtt_u_str), domain=mesh,t=t,degree=6)


    f00=dtt_u_ex0+div(grad(div(grad(u_ex0))))-VK(u_ex0,v_ex0)
    f11=dtt_u_ex1+div(grad(div(grad(u_ex0))))-VK(u_ex1,v_ex1)

    g00=div(grad(div(grad(v_ex0))))+0.5*VK(u_ex0,u_ex0)
    g11=div(grad(div(grad(v_ex1))))+0.5*VK(u_ex1,u_ex1)

    u0=interpolate(u_ex0,Hh.sub(0).collapse())
    v0=interpolate(v_ex0,Hh.sub(1).collapse())
    def rit_projection(W,data):
        thet_proj_sol = Function(W)
        thet_projection = TrialFunction(W)
        ph_projection = TestFunction(W)
        AAAA = stiffness(thet_projection,ph_projection)

        L= -0.5*inner(VK(data,data),ph_projection)*dx+g00*ph_projection*dx
       
        solve(AAAA == L, thet_proj_sol,DirichletBC(W,v_ex0,"on_boundary"), solver_parameters={'linear_solver':'mumps'})
        return thet_proj_sol
 #   v0=rit_projection(Hh.sub(1).collapse(),u_ex0)

    u1=interpolate(u_ex1,Hh.sub(0).collapse())
   
    v1=interpolate(v_ex1,Hh.sub(1).collapse())
    bcup = DirichletBC(Hh.sub(0), u_ex1, "on_boundary")
    bcvp = DirichletBC(Hh.sub(1), v_ex1, "on_boundary")
    bcHp = [bcup,bcvp]
 
 
    solp = Function(Hh); dSolp = TrialFunction(Hh)
    up,vp = split(solp)
    phip,psip = TestFunctions(Hh)
    AA = (2.0 / dt**2) * ((up - u0 - dt * u_ex0) * phip * dx)\
        +0.5*stiffness(up+u0,phip)+stiffness(vp,psip)\
        -0.5*VK(up,vp)*phip*dx+0.5*VK(up,up)*phip*dx\
        +0.5*VK(up,up)*psip*dx\
        -0.5*(f00+f11)*phip*dx-g11*psip*dx
          
    Tang = derivative(AA,solp,dSolp)
       
    solve(AA == 0, solp, bcHp, J=Tang, \
                solver_parameters={"newton_solver":{"linear_solver":'mumps', "relative_tolerance": 1e-8}})
  #  u1,v1 = solp.split() 

    Es_0=0.0#dt*pow(assemble(((1./dt*(u_ex1-u_ex0-u1+u0))**2)*dx),0.5)
    Es_H2 =0.0#dt*energynorm(0.5*(-u0-u1+u_ex1+u_ex0))

    Esv_0 =0.0#max(dt*pow(assemble(((v_ex0-v0)**2)*dx),0.5),dt*pow(assemble(((v_ex1-v1)**2)*dx),0.5))
    Esv_H2=0.0#max(dt*energynorm(v_ex0-v0),dt*energynorm(v_ex1-v1))
 #while (t <= Tfinal):
    while (t < Tfinal):

        if t + dt > Tfinal:
            dt = Tfinal - t
            t  = Tfinal
        else:
            t += dt
        print("t=%.4f" % t)
        u_exact0 = Expression(str2exp(u_str), domain=mesh,t=t-2*dt,dt=dt,degree=6) 
        u_exact1 = Expression(str2exp(u_str), domain=mesh,t=t-dt,dt=dt,degree=6)
        u_exact2 = Expression(str2exp(u_str), domain=mesh,t=t,dt=dt,degree=6)
        
        v_exact0 = Expression(str2exp(v_str), domain=mesh,t=t-2*dt,dt=dt,degree=6) 
        v_exact1 = Expression(str2exp(v_str), domain=mesh,t=t-dt,dt=dt,degree=6)
        v_exact2 = Expression(str2exp(v_str), domain=mesh,t=t,dt=dt,degree=6)

        dtt_u_exact0 = Expression(str2exp(dtt_u_str), domain=mesh,t=t-2*dt,dt=dt,degree=6) 
        dtt_u_exact1 = Expression(str2exp(dtt_u_str), domain=mesh,t=t-dt,dt=dt,degree=6)
        dtt_u_exact2 = Expression(str2exp(dtt_u_str), domain=mesh,t=t,dt=dt,degree=6)
        
        f0=dtt_u_exact0+div(grad(div(grad(u_exact0))))-VK(u_exact0,v_exact0)
        f1=dtt_u_exact1+div(grad(div(grad(u_exact1))))-VK(u_exact1,v_exact1)
        f2=dtt_u_exact2+div(grad(div(grad(u_exact2))))-VK(u_exact2,v_exact2)
             
        g0=div(grad(div(grad(v_exact0))))+0.5*VK(u_exact0,u_exact0)
        g1=div(grad(div(grad(v_exact1))))+0.5*VK(u_exact1,u_exact1)
        g2=div(grad(div(grad(v_exact2))))+0.5*VK(u_exact2,u_exact2)
        
        bcu = DirichletBC(Hh.sub(0), u_exact2, "on_boundary")
        bcv = DirichletBC(Hh.sub(1), v_exact2, "on_boundary")
        bcH = [bcu,bcv]
              
        sol = Function(Hh); dSol = TrialFunction(Hh)
        u,v = split(sol)
        phi,psi = TestFunctions(Hh)
        # Define+ stif(u+2*u1+u0,phi) the expression
        
        FF =(1/dt**2)*(u-2*u1+u0)*phi*dx\
             + 1/4*stiffness(u+2*u1+u0,phi)\
             -1./4*(VK(u,v)*phi)*dx-1./4*(VK(u1,v1)*phi)*dx-1./4*(VK(u0,v0)*phi)*dx\
             + stiffness(v,psi)+1./2*inner(VK(u,u),psi)*dx\
             - 1./4*(f0+2*f1+f2)*phi*dx-(g2)*psi*dx
        
        Tang = derivative(FF,sol,dSol)
        
        solve(FF == 0, sol, bcH, J=Tang, \
                solver_parameters={"newton_solver":{"linear_solver":'mumps', "relative_tolerance": 1e-8}})
        u_h,v_h = sol.split()
        
        #dt_uexact=1./dt*(u_exact-u_eaxct2)
        dt_uh=1./dt*(u_h-u1)
      #  u_exx=1./2*(u_exact2+u_exact)
        u_hh=1./2*(u_h+u1)
        u_h.rename("u_h","u_h"); file_vk.write(u_h, t)
        v_h.rename("v_h","v_h"); file_vk.write(v_h, t)

        #u_exact=Expression(str2exp(u_str), domain=mesh,t=t+dt,dt=dt,degree=6)
        #v_exact=Expression(str2exp(v_str), domain=mesh,t=t+dt,dt=dt,degree=6)
        dt_uexact2 = 1./dt*(u_exact2-u_exact1)
        #Errors
        u_exx=0.5*(u_exact2+u_exact1)       
        #vex_mid=0.5*(v_exact2+v_exact1)    
       # vh_mid=1./2*(v_h+v1)
        Es_0=max(Es_0,dt*pow(assemble(((dt_uexact2-dt_uh)**2)*dx),0.5))
        Es_H2=max(Es_H2,dt*energynorm(u_exx-u_hh))

        Esv_0=max(Esv_0,dt*pow(assemble(((v_exact2-v_h)**2)*dx),0.5))
        Esv_H2=max(Esv_H2,dt*energynorm(v_exact2-v_h))     
        
        assign(u0,u1)
        assign(u1,u_h)
        assign(v0,v1)
        assign(v1,v_h)
       

    es_0.append(Es_0)
    es_H2.append(Es_H2)
    esv_0.append(Esv_0)
    esv_H2.append(Esv_H2)
    #Rates of convergence
    if(nk>0):
        rs_0.append(ln(es_0[nk]/es_0[nk-1])/(ln(dtvec[nk]/dtvec[nk-1])))
        rs_H2.append(ln(es_H2[nk]/es_H2[nk-1])/(ln(dtvec[nk]/dtvec[nk-1])))
        rsv_0.append(ln(esv_0[nk]/esv_0[nk-1])/ln(dtvec[nk]/dtvec[nk-1]))
        rsv_H2.append(ln(esv_H2[nk]/esv_H2[nk-1])/ln(dtvec[nk]/dtvec[nk-1]))


# ********  Generating error history **** #
print('==================================================================================================================')
print('   $h$  &  Eu(0)   &  Ru(0) &   Eu(2)   &  Ru(2) &  Ev(0)   &  Rv(0) &   Ev(2)   &  Rv(2) ')
print('==================================================================================================================')
for nk in range(nkmax):
    print(' {:.3f} & {:6.2e} & {:.3f}  & {:6.2e} & {:.3f}& {:6.2e} & {:.3f}  & {:6.2e} & {:.3f}'.format( dtvec[nk], es_0[nk], rs_0[nk], es_H2[nk], rs_H2[nk], esv_0[nk], rsv_0[nk], esv_H2[nk], rsv_H2[nk]))
print('==================================================================================================================')

