import unit_renorm, torch, math
B,H,L,D = 1,2,4,8
out = unit_renorm.run(B,H,L,D)       # [B,L,H,D] on CPU
alpha_expected = math.exp(-1)*0.5
print(out.unique())                  # every element should be ≈ alpha_expected