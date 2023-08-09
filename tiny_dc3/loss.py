def soft_loss(objective, eq_resid, ineq_resid, lamb_h=10, lamb_g=10, int_mask=None, lambg_z=10, pow=2):

    def loss(y):
        obj = objective(y).pow(pow).mean()
        h = lamb_h*sum(h(y).abs().pow(pow) for h in eq_resid).mean()
        g = lamb_g*sum(g(y).relu().pow(pow) for g in ineq_resid).mean()

        if int_mask is None:
            return obj + h + g
        
        # z = 
    
    return loss