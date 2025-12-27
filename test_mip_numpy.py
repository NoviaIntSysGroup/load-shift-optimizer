import numpy as np
from mip import Model, MINIMIZE, CBC

def test_mip_numpy():
    m = Model(solver_name=CBC)
    x = m.add_var()
    val = np.float64(10.0)
    
    print(f"Type of val: {type(val)}")
    
    expr1 = (val == x)
    print(f"Type of (val == x): {type(expr1)}")
    
    expr2 = (x == val)
    print(f"Type of (x == val): {type(expr2)}")

    try:
        m += (val == x)
        print("m += (val == x) worked")
    except Exception as e:
        print(f"m += (val == x) failed: {e}")

    try:
        m += (x == val)
        print("m += (x == val) worked")
    except Exception as e:
        print(f"m += (x == val) failed: {e}")

if __name__ == "__main__":
    test_mip_numpy()
