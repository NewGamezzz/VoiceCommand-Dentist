import numpy as np

def edit_distance(a, b):
  a = a.replace(' ', '')
  b = b.replace(' ', '')
  len_a, len_b = len(a), len(b)
  dp = np.zeros((len_a+1, len_b+1))
  
  for i in range(len_a+1): # <== always insert
    dp[i, 0] = i
  
  for j in range(len_b+1): # <== always insert
    dp[0, j] = j

  for i in range(1, len_a+1):
    for j in range(1, len_b+1):
      if a[i-1] == b[j-1]:
        dp[i, j] = dp[i-1, j-1]
      
      else:
        dp[i, j] = 1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])

  return dp[len_a, len_b]
