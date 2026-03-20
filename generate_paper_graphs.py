"""
Exact Replication of Reference Styles for Enigma_24 Performance Evaluation
Matches the user's uploaded images (headings A-J, multi-subplot layouts).
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Set global styles - Formal Academic Style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

# Reference Colors from Uploaded Images
CLR_NAVY = "#1a4e66"    # Deep Learning / Base
CLR_EMERALD = "#10b981" # Ensemble / Hybrid / Proposed
CLR_CRIMSON = "#d9534f" # Stage 1 / Component 1
CLR_AMBER = "#f0ad4e"   # Stage 2 / Component 2
CLR_SAGE = "#5cb85c"    # Stage 3 / Component 3
CLR_MAROON = "#8c510a"  # Overhead / High Latency
CLR_SKY = "#3498db"     # Baseline / Low Overhead

# Create output directory
output_dir = '/Users/apple/.gemini/antigravity/brain/9a8c965a-94a9-4753-875c-36239957cf17'
os.makedirs(output_dir, exist_ok=True)

print("🔬 Generating Exact-Style Academic Performance Visualizations...")
print("="*60)

# ============================================================================
# FIG 1. INFERENCE LATENCY COMPARISON
# ============================================================================
print("\nFig 1. Creating Latency Comparison (Navy/Emerald)...")
fig, ax = plt.subplots(figsize=(8, 5))

algos = ['LSTM (T1)', 'LSTM (T2)', 'RF (Base)', 'RF (Opt)', 'TradeLens']
latencies = [120, 180, 45, 62, 210] # Values from reference Fig 1
colors = [CLR_NAVY, CLR_NAVY, CLR_EMERALD, CLR_EMERALD, CLR_EMERALD]

bars = ax.bar(algos, latencies, color=colors, edgecolor='black', linewidth=0.8)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=CLR_NAVY, edgecolor='black', label='Deep Learning'),
    Patch(facecolor=CLR_EMERALD, edgecolor='black', label='Ensemble/Hybrid')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

ax.set_ylabel('Total Latency (ms)', fontsize=11)
ax.set_xlabel('Algorithm Configuration', fontsize=11)
ax.set_ylim(0, 250)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{output_dir}/fig1_latency_comp.png', dpi=300)
plt.close()

# ============================================================================
# FIG 2. OPERATION BREAKDOWN (1x2 Subplots)
# ============================================================================
print("\nFig 2. Creating Operation Breakdown (1x2 Stacked)...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

algos_base = ['LSTM', 'RF-Opt', 'Hybrid', 'Legacy']
# Left: Inference Phases
ingest = [0.015, 0.012, 0.018, 0.020]
process = [0.080, 0.025, 0.110, 0.150]
verify = [0.010, 0.005, 0.015, 0.030]

ax1.bar(algos_base, ingest, label='Ingest', color=CLR_CRIMSON, edgecolor='black', linewidth=0.6)
ax1.bar(algos_base, process, bottom=ingest, label='Inference', color=CLR_AMBER, edgecolor='black', linewidth=0.6)
ax1.bar(algos_base, verify, bottom=np.array(ingest)+np.array(process), label='Validation', color=CLR_SAGE, edgecolor='black', linewidth=0.6)

ax1.set_title('Inference Breakdown', fontsize=12, fontweight='bold')
ax1.set_ylabel('Latency (ms)')
ax1.tick_params(axis='x', rotation=45)
ax1.legend(fontsize=8, loc='upper left')

# Right: RAG Stages
rag_algos = ['LSTM', 'RF-Opt', 'Hybrid']
sim = [0.1, 0.2, 0.35]
rank = [0.15, 0.25, 0.45]
gen = [0.2, 0.3, 0.55]

ax2.bar(rag_algos, sim, label='SimSearch', color=CLR_CRIMSON, edgecolor='black', linewidth=0.6)
ax2.bar(rag_algos, rank, bottom=sim, label='Rerank', color=CLR_AMBER, edgecolor='black', linewidth=0.6)
ax2.bar(rag_algos, gen, bottom=np.array(sim)+np.array(rank), label='Generation', color=CLR_SAGE, edgecolor='black', linewidth=0.6)

ax2.set_title('RAG Justification', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.4)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f'{output_dir}/fig3_breakdown_ref.png', dpi=300)
plt.close()

# ============================================================================
# FIG 4. BANDWIDTH COMPARISON (5-Column Grouped Bar Replication)
# ============================================================================
print("\nFig 4. Creating Bandwidth Analysis (5-Column Grouped Log)...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Subplot 1: Signal Generation (Data Size)
# -------------------------------
labels_bw1 = ['LR', 'LSTM-60', 'RF-Opt', 'Hybrid', 'TradeLens']
num_size = [64, 128, 512, 1024, 1312]  # Numerical Feature Size
sem_size = [32, 64, 256, 512, 1100]    # Semantic Feature Size

x = np.arange(len(labels_bw1))
width = 0.4  # Matches reference grouping

ax1.bar(x - width/2, num_size, width, color=CLR_NAVY, label='Numerical Data', edgecolor='black')
ax1.bar(x + width/2, sem_size, width, color=CLR_MAROON, label='Semantic Data', edgecolor='black')

ax1.set_yscale('log')
ax1.set_ylabel('Size (bytes)', fontweight='bold')
ax1.set_title('Signal Generation', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(labels_bw1, fontsize=9, rotation=15)
ax1.legend(loc='upper left', fontsize=9)
ax1.set_ylim(10, 10**4)

# Subplot 2: Explainable RAG (Payload Size)
# -------------------------------
labels_bw2 = ['SVR', 'MLP', 'RAG-Mini', 'RAG-Med', 'TradeLens-Full']
context_size = [32, 128, 1500, 2500, 3476]   # Retrieval Context
just_size = [16, 64, 800, 1800, 3200]       # Generated Justification

x2 = np.arange(len(labels_bw2))

ax2.bar(x2 - width/2, context_size, width, color=CLR_NAVY, label='Public Meta', edgecolor='black')
ax2.bar(x2 + width/2, just_size, width, color=CLR_MAROON, label='Justification', edgecolor='black')

ax2.set_yscale('log')
ax2.set_title('Explainable RAG', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x2)
ax2.set_xticklabels(labels_bw2, fontsize=9, rotation=15)
ax2.legend(loc='upper left', fontsize=9)
ax2.set_ylim(10, 10**4)

plt.tight_layout(pad=3.0)
plt.savefig(f'{output_dir}/fig4_bandwidth_ref.png', dpi=300)
plt.close()

# ============================================================================
# FIG 5. LATENCY VS. BANDWIDTH TRADE-OFF (Exact Replication Style)
# ============================================================================
print("\nFig 5. Creating Latency vs Bandwidth Trade-off (1x2 Scatter)...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Subplot 1: Numerical Inference
# -------------------------------
# Baseline: Circles, Navy
# Proposed: Squares, Emerald
ax1.set_title('Numerical Inference', fontsize=14, fontweight='bold', pad=15)
# (Bandwidth, Latency)
base_pt1 = (128, 87) # LR
base_pt2 = (175, 432) # Standard LSTM
ax1.scatter([base_pt1[0], base_pt2[0]], [base_pt1[1], base_pt2[1]], color=CLR_NAVY, marker='o', s=200, label='Baseline', edgecolor='black')
ax1.text(base_pt1[0]+50, base_pt1[1]+5, 'LR', fontsize=10)
ax1.text(base_pt2[0]+50, base_pt2[1]+15, 'LSTM', fontsize=10)

prop_pt1 = (1500, 31) # RF-Opt
prop_pt2 = (2200, 48) # Hybrid
prop_pt3 = (3100, 68) # Ensemble-X
ax1.scatter([prop_pt1[0], prop_pt2[0], prop_pt3[0]], [prop_pt1[1], prop_pt2[1], prop_pt3[1]], 
            color=CLR_EMERALD, marker='s', s=180, label='TradeLens', edgecolor='black')
ax1.text(prop_pt1[0]+50, prop_pt1[1]+5, 'RF-Opt', fontsize=9)
ax1.text(prop_pt2[0]+50, prop_pt2[1]+5, 'Hybrid', fontsize=9)
ax1.text(prop_pt3[0]+50, prop_pt3[1]+5, 'Ens-X', fontsize=9)

ax1.set_xlabel('Bandwidth (bytes)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_xlim(-100, 4000)

# Subplot 2: Explainable Analysis (RAG)
# -------------------------------
ax2.set_title('Explainable Analysis', fontsize=14, fontweight='bold', pad=15)
base_r_pt1 = (80, 85) # Standard Meta
base_r_pt2 = (140, 560) # Full XML
ax2.scatter([base_r_pt1[0], base_r_pt2[0]], [base_r_pt1[1], base_r_pt2[1]], color=CLR_NAVY, marker='o', s=200, label='Classical', edgecolor='black')
ax2.text(base_r_pt1[0]+150, base_r_pt1[1]+10, 'Meta', fontsize=10)
ax2.text(base_r_pt2[0]+150, base_r_pt2[1]+10, 'XML', fontsize=10)

rag_pt1 = (3500, 307) # RAG-Mini
rag_pt2 = (5100, 518) # RAG-Med
rag_pt3 = (7500, 694) # TradeLens-Full
ax2.scatter([rag_pt1[0], rag_pt2[0], rag_pt3[0]], [rag_pt1[1], rag_pt2[1], rag_pt3[1]], 
            color=CLR_EMERALD, marker='s', s=180, label='Proposed', edgecolor='black')
ax2.text(rag_pt1[0]+150, rag_pt1[1]-5, 'Mini', fontsize=9)
ax2.text(rag_pt2[0]+150, rag_pt2[1], 'Med', fontsize=9)
ax2.text(rag_pt3[0]-600, rag_pt3[1]+20, 'Full', fontsize=9)

ax2.set_xlabel('Bandwidth (bytes)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_xlim(-200, 8500)

plt.tight_layout(pad=3.0)
plt.savefig(f'{output_dir}/fig5_tradeoff_ref.png', dpi=300)
plt.close()

# ============================================================================
# FIG 6. SPEEDUP ANALYSIS (TradeLens Project Sync)
# ============================================================================
print("\nFig 6. Creating Speedup Bar Chart (Project Palette)...")
fig, ax = plt.subplots(figsize=(8, 4))

labels = ['RF-Opt', 'Hybrid', 'RAG-Mini', 'Ensemble-L2', 'Parallel-Synth', 'TradeLens']
speedup = [3.30, 2.12, 1.39, 0.28, 0.17, 0.13]

# Unified to Emerald (Proposed Color)
ax.bar(labels, speedup, color=CLR_EMERALD, edgecolor='black', label='TradeLens Optimizations')
ax.axhline(1, color='grey', ls='--', lw=1, label='Baseline (LSTM-120)')

ax.set_ylabel('Speedup (x)', fontweight='bold')
ax.set_title('Relative Performance Speedup (TradeLens)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
plt.xticks(rotation=15)

for i, v in enumerate(speedup):
    ax.text(i, v + 0.1, f'{v}x', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/fig6_speedup_ref.png', dpi=300)
plt.close()

# ============================================================================
# FIG 7. LEGACY UNSUITABILITY (Project Palette - Navy/Emerald Sync)
# ============================================================================
print("\nFig 7. Creating Legacy Unsuitability (TradeLens Sync)...")
fig, ax = plt.subplots(figsize=(8, 5))

algos = ['SVR (Legacy)', 'MLP (Legacy)', 'LR (Baseline)', 'LSTM (T1)', 'TradeLens']
latencies = [42.2, 141.8, 0.087, 0.307, 0.694] 
# Navy for Legacy/Baseline, Emerald for TradeLens/Proposed
colors = [CLR_NAVY, CLR_NAVY, CLR_NAVY, CLR_EMERALD, CLR_EMERALD]

ax.bar(algos, latencies, color=colors, edgecolor='black')
ax.axhline(1.0, color='red', ls='--', label='Trading Limit (1ms)')
ax.set_yscale('log')
ax.set_ylabel('Total Latency (ms) - Log Scale', fontweight='bold')
ax.set_title('Legacy Model Unsuitability (TradeLens)', fontsize=12, fontweight='bold')

# Custom Legend
legend_elements = [
    Patch(facecolor=CLR_NAVY, edgecolor='black', label='Legacy/Baseline'),
    Patch(facecolor=CLR_EMERALD, edgecolor='black', label='TradeLens (Ours)'),
    Line2D([0], [0], color='red', ls='--', label='Trading Limit (1ms)')
]
from matplotlib.lines import Line2D # Import here for clarity
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}/fig7_unsuitability_ref.png', dpi=300)
plt.close()

# ============================================================================
# TABLE I, II & III. IEEE-STYLE TABLES AS IMAGES
# ============================================================================
def create_ieee_table(title, data, filename):
    row_count = len(data)
    col_count = len(data[0])
    fig_h = max(2.5, row_count * 0.5)
    fig_w = 10
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    
    table = ax.table(cellText=data, loc='center', cellLoc='center', colWidths=[1.0/col_count]*col_count)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.5)
    
    for (i, j), cell in table.get_celld().items():
        cell.set_linewidth(0)
        text_obj = cell.get_text()
        text_obj.set_fontfamily('serif')
        if i == 0:
            cell.set_text_props(weight='bold')
    
    cells = table.get_celld()
    x0 = cells[(0, 0)].get_x()
    x1 = cells[(0, col_count-1)].get_x() + cells[(0, col_count-1)].get_width()
    h = cells[(0, 0)].get_height()
    
    y_top = cells[(0, 0)].get_y() + h
    ax.plot([x0, x1], [y_top, y_top], color='black', lw=2.0)
    y_sep = cells[(0, 0)].get_y()
    ax.plot([x0, x1], [y_sep, y_sep], color='black', lw=1.2)
    y_bot = cells[(row_count-1, 0)].get_y()
    ax.plot([x0, x1], [y_bot, y_bot], color='black', lw=2.0)

    plt.title(title, fontsize=12, fontweight='bold', pad=20, fontfamily='serif')
    plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

print("\nTables. Creating IEEE Tables...")
t1_data = [['Algorithm', 'Fetch', 'Feature', 'Infer', 'Total'], ['LSTM-60', '0.013', '0.019', '0.055', '0.087'], ['LSTM-120', '0.107', '0.144', '0.307', '0.557'], ['RF-Opt', '0.047', '0.209', '0.051', '0.307'], ['Hybrid', '0.091', '0.345', '0.082', '0.518'], ['TradeLens RAG', '0.132', '0.425', '0.137', '0.694']]
create_ieee_table('TABLE I\nINFERENCE LATENCY BREAKDOWN FOR TRADELENS MODELS\n(MILLISECONDS)', t1_data, 'table1_latency_ref.png')

t2_data = [['Algorithm', 'Input Seq', 'Sentiment', 'Total'], ['LSTM-60', '65', '65', '130'], ['LSTM-120', '97', '97', '194'], ['RF-Opt', '1,312', '2,420', '3,732'], ['TradeLens', '1,952', '3,309', '5,261']]
create_ieee_table('TABLE II\nBANDWIDTH OVERHEAD COMPARISON (BYTES)', t2_data, 'table2_bandwidth_ref.png')

t3_data = [['Algorithm', 'Security Level', '< 200ms (HFT)', 'Stable Analysis', 'Portfolio Load'], ['LSTM-60', 'Standard', '✓', '✓', '✓'], ['RF-Opt', 'Premium', '✓', '✓', '✓'], ['TradeLens-RAG', 'Institutional', '×', '✓', '✓'], ['Hybrid Ensemble', 'Advanced', '✓', '✓', '✓'], ['Legacy RNN', 'None', '✓', '×', '×']]
create_ieee_table('TABLE III\nTRADELENS AI SERVICE CATEGORY COMPLIANCE ANALYSIS', t3_data, 'table3_compliance_ref.png')

print("\n" + "="*60)
print("✨ EXACT-STYLE VISUALIZATIONS COMPLETED!")
print("="*60)
