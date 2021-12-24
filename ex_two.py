import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ex_one import *

def cof_fx(Dx=1.7*10**(-5),r1=1.5,r2=3.5,r23=0.1,r3=5.,R1=0.03,R2=0.07,R3=0.15,R0=0.05):
    x=np.zeros((12,12))
    y=np.zeros((12,1))
    x[0][1]=1;y[0][0]=0
    x[1][3]=1;y[1][0]=0
    x[2][0]=R1;x[2][4]=-R1;x[2][5]=-1/R1**2;y[2][0]=0
    x[3][2]=R1**2.;x[3][6]=-R1**2.;x[3][7]=-1/R1**3.;y[3][0]=0
    x[4][0]=1/r1;x[4][4]=-1/r2;x[4][5]=2/(r2*R1**3);y[4][0]=(Dx/(4*np.pi*R0**3))*(1-r2/r1)
    x[5][2]=2*R1/r1;x[5][6]=-2*R1/r2;x[5][7]=3/(r2*R1**4);y[5][0]=((2*Dx*R1)/(4*np.pi*R0**4))*(1-r2/r1)
    x[6][4]=1/r2;x[6][5]=-2/(r2*R2**3);x[6][8]=-1/r3;x[6][9]=2/(r3*R2**3);y[6][0]=((-2*Dx)/(4*np.pi*R2**3))*(r2/r3-1)
    x[7][6]=2*R2/r2;x[7][7]=-3/(r2*R2**4);x[7][10]=-2*R2/r3;x[7][11]=3/(r3*R2**4);y[7][0]=((-3*Dx*R0)/(4*np.pi*R2**4))*(r2/r3-1)
    x[8][8]=1;x[8][9]=-2/R3**3;y[8][0]=(2*Dx*r2)/(4*np.pi*R3**3)
    x[9][10]=2*R3;x[9][11]=-3/R3**4;y[9][0]=(3*Dx*r2*R0)/(4*np.pi*R3**4)
    x[10][4]=R2+r23/r2;x[10][5]=1/R2**2-(2*r23)/(r2*R2**3);x[10][8]=-R2;x[10][9]=-1/R2**2;y[10][0]=(2*Dx*r23)/(4*np.pi*R2**3)
    x[11][6]=R2**2+(2*r23*R2)/r2;x[11][7]=1/R2**3-3*r23/(r2*R2**4);x[11][10]=-R2**2;x[11][11]=-1/R2**3;y[11][0]=(3*Dx*r23*R0)/(4*np.pi*R2**4)
    a = np.array(x)
    b=np.array(y)
    d=np.linalg.det(a)
    n=a.shape[0]
    ans=[]
    for i in range(n):
        x=a.copy()
        for j in range(n):
            x[j][i]=b[j][0]
        x=np.linalg.det(x)
        ans.append(x/d)
    return ans

def cof_fy(Dy=1.7*10**(-5),r1=1.5,r2=3.5,r23=0.1,r3=5.,R1=0.03,R2=0.07,R3=0.15,R0=0.05):
    x=np.zeros((12,12))
    y=np.zeros((12,1))
    x[0][1]=1;y[0][0]=0
    x[1][3]=1;y[1][0]=0
    x[2][0]=R1;x[2][4]=-R1;x[2][5]=-1/R1**2;y[2][0]=0
    x[3][2]=R1**2.;x[3][6]=-R1**2.;x[3][7]=-1/R1**3.;y[3][0]=0
    x[4][0]=1/r1;x[4][4]=-1/r2;x[4][5]=2/(r2*R1**3);y[4][0]=(Dy/(4*np.pi*R0**3))*(1-r2/r1)
    x[5][2]=2*R1/r1;x[5][6]=-2*R1/r2;x[5][7]=3/(r2*R1**4);y[5][0]=((2*Dy*R1)/(4*np.pi*R0**4))*(1-r2/r1)
    x[6][4]=1/r2;x[6][5]=-2/(r2*R2**3);x[6][8]=-1/r3;x[6][9]=2/(r3*R2**3);y[6][0]=((-2*Dy)/(4*np.pi*R2**3))*(r2/r3-1)
    x[7][6]=2*R2/r2;x[7][7]=-3/(r2*R2**4);x[7][10]=-2*R2/r3;x[7][11]=3/(r3*R2**4);y[7][0]=((-3*Dy*R0)/(4*np.pi*R2**4))*(r2/r3-1)
    x[8][8]=1;x[8][9]=-2/R3**3;y[8][0]=(2*Dy*r2)/(4*np.pi*R3**3)
    x[9][10]=2*R3;x[9][11]=-3/R3**4;y[9][0]=(3*Dy*r2*R0)/(4*np.pi*R3**4)
    x[10][4]=R2+r23/r2;x[10][5]=1/R2**2-(2*r23)/(r2*R2**3);x[10][8]=-R2;x[10][9]=-1/R2**2;y[10][0]=(2*Dy*r23)/(4*np.pi*R2**3)
    x[11][6]=R2**2+(2*r23*R2)/r2;x[11][7]=1/R2**3-3*r23/(r2*R2**4);x[11][10]=-R2**2;x[11][11]=-1/R2**3;y[11][0]=(3*Dy*r23*R0)/(4*np.pi*R2**4)
    a = np.array(x)
    b=np.array(y)
    d=np.linalg.det(a)
    n=a.shape[0]
    ans=[]
    for i in range(n):
        x=a.copy()
        for j in range(n):
            x[j][i]=b[j][0]
        x=np.linalg.det(x)
        ans.append(x/d)
    return ans

def cof_fz(Dz=0.3*10**(-5),r1=1.5,r2=3.5,r23=0.1,r3=5.,R1=0.03,R2=0.07,R3=0.15,R0=0.05):
    x=np.zeros((12,12))
    y=np.zeros((12,1))
    x[0][1]=1;y[0][0]=0
    x[1][3]=1;y[1][0]=0
    x[2][0]=R1;x[2][4]=-R1;x[2][5]=-1/R1**2;y[2][0]=0
    x[3][2]=R1**2.;x[3][6]=-1*R1**2.;x[3][7]=-1/R1**3;y[3][0]=0
    x[4][0]=1/r1;x[4][4]=-1/r2;x[4][5]=2/(r2*R1**3);y[4][0]=((-2*Dz)/(4*np.pi*R0**3))*(1-r2/r1)
    x[5][2]=2*R1/r1;x[5][6]=-2*R1/r2;x[5][7]=3/(r2*R1**4);y[5][0]=((-6*Dz*R1)/(4*np.pi*R0**4))*(1-r2/r1)
    x[6][4]=1/r2;x[6][5]=-2/(r2*R2**3);x[6][8]=-1/r3;x[6][9]=2/(r3*R2**3);y[6][0]=((-2*Dz)/(4*np.pi*R2**3))*(r2/r3-1)
    x[7][6]=2*R2/r2;x[7][7]=-3/(r2*R2**4);x[7][10]=-2*R2/r3;x[7][11]=3/(r3*R2**4);y[7][0]=((-6*Dz*R0)/(4*np.pi*R2**4))*(r2/r3-1)
    x[8][8]=1;x[8][9]=-2/R3**3;y[8][0]=(2*Dz*r2)/(4*np.pi*R3**3)
    x[9][10]=2*R3;x[9][11]=-3/R3**4;y[9][0]=(6*Dz*r2*R0)/(4*np.pi*R3**4)
    x[10][4]=R2+r23/r2;x[10][5]=1/R2**2-(2*r23)/(r2*R2**3);x[10][8]=-R2;x[10][9]=-1/R2**2;y[10][0]=(2*Dz*r23)/(4*np.pi*R2**3)
    x[11][6]=R2**2+(2*r23*R2)/r2;x[11][7]=1/R2**3-3*r23/(r2*R2**4);x[11][10]=-R2**2;x[11][11]=-1/R2**3;y[11][0]=(6*Dz*r23*R0)/(4*np.pi*R2**4)
    a = np.array(x)
    b=np.array(y)
    d=np.linalg.det(a)
    n=a.shape[0]
    ans=[]
    for i in range(n):
        x=a.copy()
        for j in range(n):
            x[j][i]=b[j][0]
        x=np.linalg.det(x)
        ans.append(x/d)
    return ans

def f (Dx=1.7*10**(-5),Dy=1.7*10**(-5),Dz=0.3*10**(-5),r1=1.5,r2=3.5,r23=0.1,r3=5.,R1=0.03,R2=0.07,R3=0.15,R0=0.05,tet=np.pi/2,psi=np.pi/6):
    cofx=cof_fx(Dx,r1,r2,r23,r3,R1,R2,R3,R0)
    cofy=cof_fy(Dy,r1,r2,r23,r3,R1,R2,R3,R0)
    cofz=cof_fz(Dz,r1,r2,r23,r3,R1,R2,R3,R0)
    fix=((Dx*r2)/(4*np.pi*R3**2)+cofx[8]*R3+cofx[9]/R3**2)*np.sin(tet)*np.cos(psi)+((Dx*r2*R0)/(4*np.pi*R3**3)+cofx[10]*R3**2+cofx[11]/R3**3)*(3/2*np.cos(2*tet))*np.cos(psi)
    fiy=((Dy*r2)/(4*np.pi*R3**2)+cofy[8]*R3+cofy[9]/R3**2)*np.sin(tet)*np.sin(psi)+((Dy*r2*R0)/(4*np.pi*R3**3)+cofy[10]*R3**2+cofy[11]/R3**3)*(3/2*np.cos(2*tet))*np.sin(psi)
    fiz=((Dz*r2)/(4*np.pi*R3**2)+cofz[8]*R3+cofz[9]/R3**2)*np.cos(tet)+((2*Dz*r2*R0)/(4*np.pi*R3**3)+cofz[10]*R3**2+cofz[11]/R3**3)*(1/4*(3*np.cos(2*tet)+1))
    return(fix+fiy+fiz)

def cardioida (D=1.7*10**(-5),T=1,B=-np.pi/2,n=1000,direction=-1,pol=0):
    t=np.linspace(0,T,n)
    a=(2*np.pi*t)/T*(direction)+pol
    Dxy=D/2*(1-np.cos(a+B))
    Dx=Dxy*np.cos(a)
    Dy=Dxy*np.sin(a)
    return Dx,Dy,t,a,Dxy

def analiz2(Dx,Dy,Dz,r1,r2,r23,r3,R1,R2,R3):
    fiL=[]
    for i in range (len(t)):
        fiL.append(f(Dx[i],Dy[i],Dz,r1,r2,r23,r3,R1[i],R2[i],R3,(R1[i]+R2[i])/2,np.pi/2,np.pi/6))
    fiL=np.array(fiL)

    fiR=[]
    for i in range (len(t)):
        fiR.append(f(Dx[i],Dy[i],Dz,r1,r2,r23,r3,R1[i],R2[i],R3,(R1[i]+R2[i])/2,np.pi/2,5*np.pi/6))
    fiR=np.array(fiR)

    fiF=[]
    for i in range (len(t)):
        fiF.append(f(Dx[i],Dy[i],Dz,r1,r2,r23,r3,R1[i],R2[i],R3,(R1[i]+R2[i])/2,np.pi/2,-1*np.pi/2))
    fiF=np.array(fiF)
    
    V1=fiL-fiR
    V2=fiF-fiR
    V3=fiF-fiL
    VL=fiL-(fiR+fiF)/2
    VR=fiR-(fiF+fiL)/2
    VF=fiF-(fiL+fiR)/2
    
    return fiL,fiR,fiF,V1,V2,V3,VL,VR,VF

def segment (a,D,ax):
    a=[a,a]
    D=[0,D]
    ax.plot(a,D,'r')

def triangle (R,ax):
    a1=[5*np.pi/6,np.pi/6]
    a2=[5*np.pi/6,-np.pi/2]
    a3=[np.pi/6,-np.pi/2]
    se=[R,R]
    ax.plot(a1,se,'g')
    ax.plot(a2,se,'g')
    ax.plot(a3,se,'g')
    ax.text(5*np.pi/6-np.pi/60,R,'RA',fontsize=12)
    ax.text(np.pi/6+np.pi/30,R-R*0.1,'LA',fontsize=12)
    ax.text(-np.pi/2+np.pi/30,R,'LL',fontsize=12)

T=1
D=1.7*10**(-5)
Dz=0.3*10**(-5)
r1=1.5
r2=3.5
r23=0.1
r3=5.
R1=0.03
R2=0.07
R3=0.15
dR=0.
B=-np.pi/2
direction=-1
n=100
D1=D

Dx,Dy,t,a,D=cardioida(D,T,B,n,direction)

fig1,ax1=plt.subplots()
ax1 = plt.axes(polar=True)
segment(a[0],D[0],ax1)
triangle(D1+D1/4,ax1)
fig1.savefig('Кардиоида0.jpg',dpi=1000)

fig1,ax1=plt.subplots()
ax1 = plt.axes(polar=True)
ax1.plot(a[0:n//4+1],D[0:n//4+1])
segment(a[n//4],D[n//4],ax1)
triangle(D1+D1/4,ax1)
fig1.savefig('Кардиоида1.jpg',dpi=1000)

fig2,ax2=plt.subplots()
ax2 = plt.axes(polar=True)
ax2.plot(a[0:n//2+1],D[0:n//2+1])
segment(a[n//2],D[n//2],ax2)
triangle(D1+D1/4,ax2)
fig2.savefig('Кардиоида2.jpg',dpi=1000)

fig3,ax3=plt.subplots()
ax3 = plt.axes(polar=True)
ax3.plot(a[0:3*n//4+1],D[0:3*n//4+1])
segment(a[3*n//4],D[3*n//4],ax3)
triangle(D1+D1/4,ax3)
fig3.savefig('Кардиоида3.jpg',dpi=1000)

fig4,ax4=plt.subplots()
ax4 = plt.axes(polar=True)
segment(a[n-1],D[n-1],ax4)
ax4.plot(a,D)
triangle(D1+D1/4,ax4)
fig4.savefig('Кардиоида.jpg',dpi=1000)

R1=R1*(1+dR*np.sin(2*np.pi*t/T))
R2=R2*(1+dR*np.sin(2*np.pi*t/T))

fiL,fiR,fiF,V1,V2,V3,VL,VR,VF=analiz2(Dx,Dy,Dz,r1,r2,r23,r3,R1,R2,R3)
for i in fiL,fiR,fiF,V1,V2,V3,VL,VR,VF:
    i*=1000

dR=0.1
R1=R1*(1+dR*np.sin(2*np.pi*t/T))
R2=R2*(1+dR*np.sin(2*np.pi*t/T))

fiL1,fiR1,fiF1,V11,V21,V31,VL1,VR1,VF1=analiz2(Dx,Dy,Dz,r1,r2,r23,r3,R1,R2,R3)
for i in fiL1,fiR1,fiF1,V11,V21,V31,VL1,VR1,VF1:
    i*=1000

fRA,fLA,fLF,df1,df2,df3,dfAVL,dfAVR,dfAVF=analiz1 (D,500*10**(-2),R3,a)
for i in fRA,fLA,fLF,df1,df2,df3,dfAVL,dfAVR,dfAVF:
    i*=1000

f1= plt.figure()
a1=f1.add_subplot()
f2= plt.figure()
a2=f2.add_subplot()
f3= plt.figure()
a3=f3.add_subplot()

fig1 = plt.figure()
ax1 = fig1.add_subplot()
fig2 = plt.figure()
ax2 = fig2.add_subplot()
fig3 = plt.figure()
ax3 = fig3.add_subplot()
fig4 = plt.figure()
ax4 = fig4.add_subplot()
fig5 = plt.figure()
ax5 = fig5.add_subplot()
fig6 = plt.figure()
ax6 = fig6.add_subplot()

a1.plot(t,fLA)
a1.plot(t,fiL)
a1.plot(t,fiL1)
a1.grid()
a1.set_xlabel('время (с)')
a1.set_ylabel('потенциал fL (мВ)')
f1.savefig('Графики зависимости потенциала L от времени.jpg',dpi=1000)

a2.plot(t,fRA)
a2.plot(t,fiR)
a2.plot(t,fiR1)
a2.grid()
a2.set_xlabel('время (с)')
a2.set_ylabel('потенциал fR (мВ)')
f2.savefig('Графики зависимости потенциала R от времени.jpg',dpi=1000)

a3.plot(t,fLF)
a3.plot(t,fiF)
a3.plot(t,fiF1)
a3.grid()
a3.set_xlabel('время (с)')
a3.set_ylabel('потенциал fF (мВ)')
f3.savefig('Графики зависимости потенциала F от времени.jpg',dpi=1000)

ax1.plot(t,df1)
ax1.plot(t,V1)
ax1.plot(t,V11)
ax1.grid()
ax1.set_xlabel('время (с)')
ax1.set_ylabel('потенциал V1 (мВ)')
fig1.savefig('потенциал V1.jpg',dpi=1000)

ax2.plot(t,df2)
ax2.plot(t,V2)
ax2.plot(t,V21)
ax2.grid()
ax2.set_xlabel('время (с)')
ax2.set_ylabel('потенциал V2 (мВ)')
fig2.savefig('потенциал V2.jpg',dpi=1000)

ax3.plot(t,df3)
ax3.plot(t,V3)
ax3.plot(t,V31)
ax3.grid()
ax3.set_xlabel('время (с)')
ax3.set_ylabel('потенциал V3 (мВ)')
fig3.savefig('потенциал V3.jpg',dpi=1000)

ax4.plot(t,dfAVL)
ax4.plot(t,VL)
ax4.plot(t,VL1)
ax4.grid()
ax4.set_xlabel('время (с)')
ax4.set_ylabel('потенциал VL (мВ)')
fig4.savefig('Графики зависимости потенциала в усиленном отведении AVL от времени.jpg',dpi=1000)

ax5.plot(t,dfAVR)
ax5.plot(t,VR)
ax5.plot(t,VR1)
ax5.grid()
ax5.set_xlabel('время (с)')
ax5.set_ylabel('потенциал VR (мВ)')
fig5.savefig('Графики зависимости потенциала в усиленном отведении AVR от времени',dpi=1000)

ax6.plot(t,dfAVF)
ax6.plot(t,VF)
ax6.plot(t,VF1)
ax6.grid()
ax6.set_xlabel('время (с)')
ax6.set_ylabel('потенциал VF (мВ)')
fig6.savefig('Графики зависимости потенциала в усиленном отведении AVF от времени.jpg',dpi=1000)

f11 = plt.figure()
a11 = f11.add_subplot()
a11.plot(t,fiR)
a11.plot(t,fiL)
a11.plot(t,fiF)
a11.grid()
a11.set_xlabel('время (с)')
a11.set_ylabel('потенциалы fRA,fLA,fLF (мВ)')
f11.savefig('2 задача fRA,fLA,fLF.jpg',dpi=1000)

f12 = plt.figure()
a12 = f12.add_subplot()
a12.plot(t,V1)
a12.plot(t,V2)
a12.plot(t,V3)
a12.grid()
a12.set_xlabel('время (с)')
a12.set_ylabel('потенциалы df1,df2,df3 (мВ)')
f12.savefig('2 задача df1,df2,df3.jpg',dpi=1000)

f13 = plt.figure()
a13 = f13.add_subplot()
a13.plot(t,VR)
a13.plot(t,VL)
a13.plot(t,VF)
a13.grid()
a13.set_xlabel('время (с)')
a13.set_ylabel('потенциалы dfAVL,dfAVR,dfAVF (мВ)')
f13.savefig('2 задача dfAVL,dfAVR,dfAVF.jpg',dpi=1000)

plt.show()