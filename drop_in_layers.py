# Framework code availible on request from UMass CS 682 staff
def drop_in_forward(x, drop_in_param):
    p, mode, random_val = drop_in_param['p'], drop_in_param['mode'], drop_in_param['random_val']
    if 'seed' in drop_in_param:
        np.random.seed(drop_in_param['seed'])

    add_mask = None
    mult_mask = None
    out = None

    if mode == 'train':
        zero_mask = (x == 0)
        non_zero_mask = (x != 0)
        # this will get the number of zeros that are in x
        num_zeros = np.count_nonzero(zero_mask)
        if num_zeros == 0:
            out = x
            mult_mask = np.ones_like(x)
        else:
            num_non_zeros = np.count_nonzero(zero_mask == 0)
            add_mask = ((np.random.rand(*x.shape) * zero_mask) < p) * random_val
            mult_mask = non_zero_mask * (num_non_zeros / ((p * num_zeros * random_val + num_non_zeros)))

            out = (x * mult_mask) + add_mask
        
    elif mode == 'test':
        out = x

    cache = (drop_in_param, add_mask, mult_mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache

def drop_in_backward(dout, cache):
    """
    Perform the backward pass for (inverted) drop_in.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from drop_in_forward.
    """
    drop_in_param, add_mask, mult_mask = cache
    mode = drop_in_param['mode']

    dx = None
    if mode == 'train':
        dx = dout * mult_mask
    elif mode == 'test':
        dx = dout
    return dx
