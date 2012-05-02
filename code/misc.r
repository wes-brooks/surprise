pdf('../figures/normal_pdf.pdf')
    x = seq(from=-3, to=3, length.out=300)
    y = dnorm(x, mean=0, sd=1/sqrt(1))
    plot(x=x, y=y, type='l', col='blue', lwd=2, bty='n', xlab='X', ylab='p(X)', main='Normal density; m=0, p=1', cex.axis=1.8, cex.lab=1.8, cex.main=2)
dev.off()


pdf('../figures/gamma_pdf.pdf')
    x = seq(from=0, to=10, length.out=1000)
    y = dgamma(x, shape=2, rate=1)
    plot(x=x, y=y, type='l', col='blue', lwd=2, bty='n', xlab='Y', ylab='p(Y)', main='Gamma density; a=2, b=1', cex.axis=1.8, cex.lab=1.8, cex.main=2)
dev.off()


pdf('../figures/t_pdf.pdf')
    x = seq(from=-3, to=3, length.out=1000)
    y = dt(x, df=4)
    plot(x=x, y=y, type='l', col='blue', lwd=2, bty='n', xlab='Z', ylab='p(Z)', main='t density; 4 degrees of freedom', cex.axis=1.8, cex.lab=1.8, cex.main=2)
dev.off()


pdf('../figures/t_pdfpost.pdf')
    x = seq(from=-3, to=3, length.out=1000)
    y = dt(x, df=7)
    plot(x=x, y=y, type='l', col='red', lwd=2, bty='n', xlab='Z', ylab='p(Z)', main='t density; 4 degrees of freedom', cex.axis=1.8, cex.lab=1.8, cex.main=2)
dev.off()


pdf('../figures/normal_pdfdraw.pdf')
    x = seq(from=-3, to=3, length.out=300)
    y = dnorm(x, mean=0, sd=1/sqrt(1))
    N = rnorm(1, mean=0, sd=1/sqrt(1))
    plot(x=x, y=y, type='l', col='blue', lwd=2, bty='n', xlab='X', ylab='p(X)', main='Normal density; m=0, p=1', cex.axis=1.8, cex.lab=1.8, cex.main=2)
dev.off()


pdf('../figures/gamma_pdfdraw.pdf')
    x = seq(from=0, to=10, length.out=1000)
    y = dgamma(x, shape=2, rate=1)
    G = rgamma(1, shape=2, rate=1)
    plot(x=x, y=y, type='l', col='blue', lwd=2, bty='n', xlab='Y', ylab='p(Y)', main='Gamma density; a=2, b=1', cex.axis=1.8, cex.lab=1.8, cex.main=2)
dev.off()


pdf('../figures/t_pdfdraws.pdf')
    x = seq(from=-3, to=3, length.out=1000)
    y = dt(x, df=4)
    T = rt(6, df=4)
    plot(x=x, y=y, type='l', col='blue', lwd=2, bty='n', xlab='Z', ylab='p(Z)', main='t density; 4 degrees of freedom', cex.axis=1.8, cex.lab=1.8, cex.main=2)
    
    for(i in 1:6) {
        lines(x=c(T[i], T[i]), y=c(0, dt(T[i], df=4)), lty=2, col='red', lwd=2)
    }
    points(x=T, y=rep(0.01, 6), cex=2, pch=20, col='red')
dev.off()


pdf('../figures/t_logpdf.pdf')
    yy = c(0, 0.4)
    x = seq(from=-5, to=10, length.out=1000)
    y1 = dt(1.1*x, df=4)
    y2 = 0.88*dt((x-1.8)/1.6, df=4)
    plot(x=x, y=y1, type='l', col='blue', lwd=2, bty='n', xlab='Z', ylab='p(Z)', main='t density; 4 degrees of freedom', ylim=yy, cex.axis=1.8, cex.lab=1.8, cex.main=2)
    par(new=TRUE)
    plot(x=x, y=y2, type='l', col='red', lwd=2, bty='n', xaxt='n', yaxt='n', ann=F, ylim=yy)
    
    yy = c(-4, 2)
    plot(x=x, y=log(y1)-log(y2), type='l', col='blue', lwd=2, bty='n', xlab='Z', ylab='p(Z)', main='t density; 4 degrees of freedom', ylim=yy, cex.axis=1.8, cex.lab=1.8, cex.main=2)
    abline(h=0)
dev.off()


pdf('../figures/t_divergence.pdf')
    yy = c(-0.3,0.5)
    plot(x=x, y=(log(y1)-log(y2))*y1, type='l', col='blue', lwd=2, bty='n', xlab='Z', ylab='p(Z)', main='t density; 4 degrees of freedom', ylim=yy, cex.axis=1.8, cex.lab=1.8, cex.main=2)
    abline(h=0, lwd=2, col='black')
dev.off()