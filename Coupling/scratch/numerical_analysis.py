import sys
sys.path.append("C:/Project/AD_Metamodeling/Coupling")
import evaluate_coupling_patient as ecp
import numpy as np

def analyze():
    spec_path = "C:/Project/AD_Metamodeling/Coupling/coupling_specs/metamodel_coupling.json"
    
    # We will redefine patients to make sure they match what we want
    patients = [
        {
            'id': 'P1', 
            'title': 'Subject 1', 
            'age': 60.0, 
            'apoe4': 0.0, 
            'amyloid_baseline': -0.5,
            'tau_baseline': 0.0,
            'amyloid_2yr': -0.2,
            'tau_2yr': 50.0,
            'zscores': [3.5, 1.0, 3.5, 0.1, 0.1, 3.5, 0.1]
        },
        {
            'id': 'P2', 
            'title': 'Subject 2', 
            'age': 65.0, 
            'apoe4': 1.0, 
            'amyloid_baseline': 15.0,
            'tau_baseline': 3.0,
            'amyloid_2yr': 25.0,
            'tau_2yr': 450.0,
            'zscores': [2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 3.5]
        }
    ]
    
    results = ecp.evaluate_all(patients, spec_path)
    
    # Let's also verify ODE outputs directly
    print("\n--- Running evaluation ---")
    
    for p in patients:
        print(f"\n======== Analyzing {p['id']} ========")
        res = results[p['id']]
        ode_s, sus_s, c_vars = res['uncoupled_ode'], res['uncoupled_sustain'], res['coupled']
        
        # Uncoupled ODE
        u_mem = ode_s[:, 5] # Assuming index 5 is memory_result_yr5
        u_clin = ode_s[:, 6] # Assuming index 6 is clinical_stage_yr5
        
        # Uncoupled SuStaIn
        u_p0 = sus_s[:, 0]
        u_p1 = sus_s[:, 1]
        u_p2 = sus_s[:, 2]
        
        print("UNCOUPLED SuStaIn Probs:")
        print(f"  Subtype 0: {np.mean(u_p0):.3f}")
        print(f"  Subtype 1: {np.mean(u_p1):.3f}")
        print(f"  Subtype 2: {np.mean(u_p2):.3f}")
        
        print("UNCOUPLED ODE Outputs:")
        print(f"  Mem Yr 5: {np.mean(u_mem):.3f}")
        print(f"  Clin Yr 5: {np.mean(u_clin):.3f} (Rounded: {np.mean(np.round(u_clin)):.3f})")
        print(f"  Tau Rate : {np.mean(ode_s[:, 0]):.3f}")
        
        # Coupled
        c_p0 = np.array(c_vars['prob_subtype_0']).flatten()
        c_p1 = np.array(c_vars['prob_subtype_1']).flatten()
        c_p2 = np.array(c_vars['prob_subtype_2']).flatten()
        
        c_mem = np.array(c_vars['memory_result_yr5']).flatten()
        c_clin = np.array(c_vars['clinical_stage_yr5']).flatten()
        c_tau = np.array(c_vars['tau_self_dynamic']).flatten()
        
        print("\nCOUPLED Outputs:")
        print(f"  Subtype 0: {np.mean(c_p0):.3f}")
        print(f"  Subtype 1: {np.mean(c_p1):.3f}")
        print(f"  Subtype 2: {np.mean(c_p2):.3f}")
        print(f"  Mem Yr 5: {np.mean(c_mem):.3f}")
        print(f"  Clin Yr 5: {np.mean(c_clin):.3f} (Rounded: {np.mean(np.round(c_clin)):.3f})")
        print(f"  Tau Rate : {np.mean(c_tau):.3f}")

if __name__ == '__main__':
    analyze()
