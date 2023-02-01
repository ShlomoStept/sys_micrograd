
from neural_network import MLP

## TEST The network out
if __name__ == '__main__':
    
    n = MLP(3, [4,4,1]) # 3 inputs to 2 layers of 4, with one output (l1 = 3,4, l2= 4,4, l3 = 4, 1 )

    xs = [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0]
        ]
    ys = [1.0, -1.0, -1.0, 1.0] # targets

    ypred = [ n(x) for x in xs ]
    
    # 5 - to imporve the network we want to calculte the MSE for each prediction to use to update the network properly

    #  A - create an automated (a) forward pass a (b) backward pass , and thenn (c) the update (gradiant decent)
    for epochs in range(20):
        
        # (a) : forward pass
        ypred = [ n(x) for x in xs ]
        loss = sum( (y_pred - y_ground_truth)**2 for y_ground_truth, y_pred in zip(ys,ypred) ) # mean_squared_error
        
        # (b) backward pass 
        # WARNING *** must reset the grads to zero, so we only accumulate the grads for this loss.backward, not prev runs
        #.      if we dont -- our steps are huge
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()
        
        # (c) the update
        # 6 - nudge the values of the weights and biases, based on the gradiant decent
        #        --> gradiant points in the direction of greater loss -> so we want to point the grad in the oppposite dir
        for p in n.parameters():
            p.data += (-0.05) * p.grad
            
            
        print(f"Epoch # {epochs}, Loss = {loss}")