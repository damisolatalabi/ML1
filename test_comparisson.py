
# test the effect of your augmentation vs no augmentation
import subprocess
import re

print("=== Augmentation Test ===")

# Backup original files
import shutil
shutil.copy2('data.py', 'data.py.backup')
shutil.copy2('train.py', 'train.py.backup')

# Test 3 things asked:
# 1. Different hidden states: 2, 3, 4, 5
# 2. Different random seeds: 3 times
# 3. Your scaling vs no scaling

results = []

for hidden_states in [2, 3, 4, 5]:
    print(f"\n--- Testing {hidden_states} hidden states ---")
    
    # Change hidden states in train.py
    with open('train.py', 'r') as f:
        train_content = f.read()
    train_content = train_content.replace('hidden_states = 3', f'hidden_states = {hidden_states}')
    with open('train.py', 'w') as f:
        f.write(train_content)
    
    for seed in [42, 123, 789]:  # 3 random seeds
        print(f"  Seed {seed}:")
        
        # Change seed in data.py
        with open('data.py', 'r') as f:
            data_content = f.read()
        
        # Find and change random.seed line
        lines = data_content.split('\n')
        for i, line in enumerate(lines):
            if 'random.seed(' in line:
                lines[i] = f'    random.seed({seed})'
                break
        
        # Test WITHOUT scaling
        print("    Without scaling: ", end='')
        for i, line in enumerate(lines):
            if 'point = [' in line and i > 100:
                # Change to NO augmentation
                lines[i] = '                            point = ['
                lines[i+1] = '                                int(i[0]),'
                lines[i+2] = '                                int(i[1]),'
                lines[i+3] = '                                int(i[2])'
                break
        
        with open('data.py', 'w') as f:
            f.write('\n'.join(lines))
        
        # Run test
        subprocess.run(['python', 'data.py'], capture_output=True)
        subprocess.run(['python', 'train.py'], capture_output=True)
        result = subprocess.run(['python', 'test.py'], capture_output=True, text=True)
        
        # Get accuracy
        acc = None
        for line in result.stdout.split('\n'):
            if 'Accuracy' in line:
                match = re.search(r'(\d+\.\d+)%', line)
                if match:
                    acc = float(match.group(1))
        
        print(f"{acc}%")
        results.append([hidden_states, seed, 'no_scaling', acc])
        
        # Test WITH your scaling
        print("    With your scaling: ", end='')
        with open('data.py', 'r') as f:
            data_content = f.read()
        
        lines = data_content.split('\n')
        for i, line in enumerate(lines):
            if 'point = [' in line and i > 100:
                # Change to YOUR scaling
                lines[i] = '                            scaling_factor = random.uniform(0.9, 1.1)'
                lines[i+1] = '                            point = ['
                lines[i+2] = '                                (int(i[0]) + np.random.normal(0, 1)) * scaling_factor,'
                lines[i+3] = '                                (int(i[1]) + np.random.normal(0, 1)) * scaling_factor,'
                lines[i+4] = '                                (int(i[2]) + np.random.normal(0, 1)) * scaling_factor'
                break
        
        with open('data.py', 'w') as f:
            f.write('\n'.join(lines))
        
        # Run test
        subprocess.run(['python', 'data.py'], capture_output=True)
        subprocess.run(['python', 'train.py'], capture_output=True)
        result = subprocess.run(['python', 'test.py'], capture_output=True, text=True)
        
        # Get accuracy
        acc = None
        for line in result.stdout.split('\n'):
            if 'Accuracy' in line:
                match = re.search(r'(\d+\.\d+)%', line)
                if match:
                    acc = float(match.group(1))
        
        print(f"{acc}%")
        results.append([hidden_states, seed, 'with_scaling', acc])

# Restore original files
shutil.copy2('data.py.backup', 'data.py')
shutil.copy2('train.py.backup', 'train.py')

print("\n=== FINAL RESULTS ===")
print("States | Seed | Method | Accuracy")
print("-" * 40)
for r in results:
    print(f"{r[0]} | {r[1]} | {r[2]} | {r[3]}%")

#keep only averages
print("\n=== AVERAGES ===")
for states in [2, 3, 4, 5]:
    no_scaling_acc = [r[3] for r in results if r[0]==states and r[2]=='no_scaling']
    scaling_acc = [r[3] for r in results if r[0]==states and r[2]=='with_scaling']
    
    avg_no = sum(no_scaling_acc)/3
    avg_with = sum(scaling_acc)/3
    
    print(f"{states} states: No scaling={avg_no:.1f}%, With scaling={avg_with:.1f}%, Improvement={avg_with-avg_no:.1f}%")