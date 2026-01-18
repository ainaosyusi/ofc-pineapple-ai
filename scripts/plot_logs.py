import re
import matplotlib.pyplot as plt

def parse_and_plot(log_path, output_path):
    steps = []
    foul_rates = []
    royalties = []
    
    with open(log_path, 'r') as f:
        content = f.read()
        
    # [Step 10000] 形式を抽出
    pattern = r"\[Step (\d+)\]\s+Games: \d+\s+Foul Rate \(last 100\): ([\d.]+)%\s+Mean Royalty \(last 100\): ([\d.]+)"
    matches = re.finditer(pattern, content)
    
    for match in matches:
        steps.append(int(match.group(1)))
        foul_rates.append(float(match.group(2)))
        royalties.append(float(match.group(3)))
        
    if not steps:
        print("No matching log patterns found.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Foul Rate (%)', color=color)
    ax1.plot(steps, foul_rates, color=color, label='Foul Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Mean Royalty', color=color)
    ax2.plot(steps, royalties, color=color, label='Mean Royalty')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('OFC Training Progress (Phase 1)')
    fig.tight_layout()
    plt.savefig(output_path)
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    parse_and_plot('logs/phase1_long_train.log', 'logs/learning_curve_phase1.png')
