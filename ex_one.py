import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def cardioida1 (D=1.7*10**(-5),T=1,B=-np.pi/2,n=1000,direction=-1,pol=0):
    t=np.linspace(0,T,n)
    a=(2*np.pi*t)/T*(direction)+pol
    Dxy=D/2*(1-np.cos(a+B))
    Dx=Dxy*np.cos(a)
    Dy=Dxy*np.sin(a)
    return Dxy,Dx,Dy,t,a

D=1.7*10**(-5)
r=500*10**(-2)
D,q,w,t,tet=cardioida1(D=D,direction=-1)
R=0.15

def analiz1 (D,r,R,tet):
    
    aRA=5*np.pi/6-tet
    aLA=np.pi/6-tet
    aLF=-np.pi/2-tet

    fRA=(r*D)/(4*np.pi*R**2)*np.cos(aRA)
    fLA=(r*D)/(4*np.pi*R**2)*np.cos(aLA)
    fLF=(r*D)/(4*np.pi*R**2)*np.cos(aLF)

    df1=fLA-fRA
    df2=fLF-fRA
    df3=fLF-fLA

    dfAVL=fLA-(fLF+fRA)/2
    dfAVR=fRA-(fLA+fLF)/2
    dfAVF=fLF-(fLA+fRA)/2
    
    return fRA,fLA,fLF,df1,df2,df3,dfAVL,dfAVR,dfAVF

fRA,fLA,fLF,df1,df2,df3,dfAVL,dfAVR,dfAVF=analiz1 (D,r,R,tet)
for i in fRA,fLA,fLF,df1,df2,df3,dfAVL,dfAVR,dfAVF:
    i*=1000


'''
f = plt.figure()
a = f.add_subplot()
a=plt.axes(polar=True)
a.plot(tet,D)
f.savefig('1 задача кардиоида.jpg',dpi=1000)
'''


f1 = plt.figure()
a1 = f1.add_subplot()
a1.plot(t,fRA)
a1.plot(t,fLA)
a1.plot(t,fLF)
a1.grid()
a1.set_xlabel('время (с)')
a1.set_ylabel('потенциалы fRA,fLA,fLF (мВ)')
f1.savefig('1 задача fRA,fLA,fLF.jpg',dpi=1000)

f2 = plt.figure()
a2 = f2.add_subplot()
a2.plot(t,df1)
a2.plot(t,df2)
a2.plot(t,df3)
a2.grid()
a2.set_xlabel('время (с)')
a2.set_ylabel('потенциалы df1,df2,df3 (мВ)')
f2.savefig('1 задача df1,df2,df3.jpg',dpi=1000)

f3 = plt.figure()
a3 = f3.add_subplot()
a3.plot(t,dfAVR)
a3.plot(t,dfAVL)
a3.plot(t,dfAVF)
a3.grid()
a3.set_xlabel('время (с)')
a3.set_ylabel('потенциалы dfAVL,dfAVR,dfAVF (мВ)')
f3.savefig('1 задача dfAVL,dfAVR,dfAVF.jpg',dpi=1000)

plt.show()