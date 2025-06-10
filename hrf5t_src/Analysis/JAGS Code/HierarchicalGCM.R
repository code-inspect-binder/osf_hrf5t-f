# JAGS implementation of a hierarchical version of the GCM

model{
  # Categorization data
  for (i in 1:nrows){
    # Each judgment represented by a categorical distribution
    y[i] ~ dcat(r[i,])
    predy[i] ~ dcat(r[i,])
  }
  
  # Determine the values of the categorical distribution, r
  for (i in 1:nrows){
    for (j in 1:ncat){
      r[i,j] = numerator[i,j]/denominator[i]
      numerator[i,j] = sum(s[stimid[i],(ntrainpercat*(j-1)+1):(ntrainpercat*(j-1)+ntrainpercat),subid[i]])^gamma
    }
    denominator[i] = sum(numerator[i,])
  }
  
  # Similarity computations, broken down by individual subject
  for (i in 1:nstim){
    for (k in 1:ntrain){
      for (sub in 1:nsubj){
          s[i,k,sub] = exp(-c[sub]*d[i,train[k]])
      }
    }
  }
  
  # Priors on parameters c, gamma. 
  # Individual subject cs are hierarchical, with chyper determining their prior's mean
  gamma ~ dunif(0,5)
  chyper ~ dunif(0,5)
  for (sub in 1:nsubj){
    # c[sub] ~ dnorm(chyper,0.1) T(0,)
    c[sub] ~ dgamma(chyper^2/4, chyper/4) #mean=chyper, sd=4
  }

}