import pylab
import matplotlib
import numpy as np
from matplotlib import pyplot as mpl
import cwt

def Threshold(obj, threshold, substitute=0):
    copy_obj = obj[:]

    for i in range( len(copy_obj) ):
        for j in range( len(copy_obj[i]) ):
            if abs(copy_obj[i][j]) < threshold:
                copy_obj[i][j] = substitute
                
    return copy_obj


def Plot_pywt(wavelist): #Takes as input the output from pywt.wavedec()
    #Establish the figure
    fig = mpl.figure()
    
    #figure out the shap of the wavelet structure
    width = len( wavelist[-1] )
    levels = len(wavelist)
    
    #initialize the mesh
    x = range(0, width)
    y = range(0, levels)
    X,Y = np.meshgrid(x,y)
    
    v=[]
    
    #plot each level
    for wavelevel in wavelist:
        steps = width / len(wavelevel)
        dx = 1 / len(wavelevel) 
        w = []
        
        val_range = max(wavelevel)-min(wavelevel)
        
        for i in range( len(wavelevel) ):
            begin = len(w)
            end = width * (i+1) / len(wavelevel)
            length = end-begin
            
            w.extend( np.repeat(wavelevel[i]/val_range, length) )
        
        v.append(w)
    
    v = np.array(v)
    mpl.pcolor(X, Y, v, cmap='jet')
        
    return v
    
def Plot_wavelets(cw):    # uses wavelets module from http://www.phy.uct.ac.za/courses/python/examples/moreexamples.html
    maxscale=4
    notes=16
    scaling="log" #or "linear"
    #scaling="linear"
    plotpower2d=False
    
    # Wavelet transform the data
    scales=cw.getscales()     
    cwt=cw.getdata()
    # power spectrum
    pwr=cw.getpower()
    scalespec=np.sum(pwr,axis=1)/scales # calculate scale spectrum
    # scales
    y=cw.fourierwl*scales
    x=np.arange(0,len(cw.getrawdata())*1.0,1.0)
    
    fig=mpl.figure(1)

    # 2-d coefficient plot
    ax=mpl.axes([0.4,0.1,0.55,0.4])
    mpl.xlabel('Time')
    plotcwt=np.clip(np.fabs(cwt.real), 0., 1000.)
    if plotpower2d: plotcwt=pwr
    im=mpl.imshow(plotcwt,cmap=mpl.cm.jet,extent=[x[0],x[-1],y[-1],y[0]],aspect='auto')
    #colorbar()
    if scaling=="log": ax.set_yscale('log')
    mpl.ylim(y[0],y[-1])
    ax.xaxis.set_ticks(np.arange(0,(len(cw.getrawdata())+1)*1.0,100.0))
    ax.yaxis.set_ticklabels(["",""])
    theposition=mpl.gca().get_position()
    
    # data plot
    ax2=mpl.axes([0.4,0.54,0.55,0.3])
    mpl.ylabel('Data')
    pos=ax.get_position()
    mpl.plot(x,cw.getrawdata(),'b-')
    mpl.xlim(0,len(cw.getrawdata())*1.0)
    ax2.xaxis.set_ticklabels(["",""])
    mpl.text(0.5,0.9,"Wavelet example with extra panes",
         fontsize=14,bbox=dict(facecolor='green',alpha=0.2),
         transform = fig.transFigure,horizontalalignment='center')

    # projected power spectrum
    ax3=mpl.axes([0.08,0.1,0.29,0.4])
    mpl.xlabel('Power')
    mpl.ylabel('Period')
    vara=100
    if scaling=="log":
        mpl.loglog(scalespec/vara+0.01,y,'b-')
    else:
        mpl.semilogx(scalespec/vara+0.01,y,'b-')
    mpl.ylim(y[0],y[-1])
    mpl.xlim(1000000000.0,10)
    
    mpl.show()

    
    

                