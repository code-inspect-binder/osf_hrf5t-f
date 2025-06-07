# JAGS implementation of a hierarchical version of the prototype model

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
      numerator[i,j] = exp(-c[subid[i]]*d[stimid[i],j])
    }
    denominator[i] = sum(numerator[i,])
  }
  
  # Priors on parameters. 
  # Individual subject cs are hierarchical, with chyper determining their prior's mean
  chyper ~ dunif(0,5)
  for (sub in 1:nsubj){
    # c[sub] ~ dnorm(chyper,0.1) T(0,)
    c[sub] ~ dgamma(chyper^2/4, chyper/4) #mean=chyper, sd=4
  }

}