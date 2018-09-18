import numpy as np
import seaborn as sns
import matplotlib
from pi_reuse import Policy
#matplotlib.use('AGG')
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style('darkgrid')

def main():

    origin_dir = 'TRAINED_POLICY/e-greedy/0-0/e-greedy-' 
    reuse_dir =  'TRAINED_POLICY/pi-reuse'

    data = dict()
    data_from_scratch = []
    reuse_policies = []

    for i in range(20):
        _, w = Policy.load(f'{origin_dir}{(i+1)*100}.npy')
        data_from_scratch.append(w)
        
    data["from-scratch"] = data_from_scratch
    for k in range(5):
        data_reuse = []
        for i in range(20):
            _, w = Policy.load(f'{reuse_dir}/0-learn-from-reuse-{k+1}/pi-reuse-{(i+1)*100}.npy')
            data_reuse.append(w)

        data[f'reuse-Pi-{k+1}'] = data_reuse
    
    ax = sns.lineplot(data=pd.DataFrame(data=data), linewidth=2.5)

    font_size = 40
    ax.ticklabel_format(style='sci', axis='x',scilimits=(0,0))
    xlocs = ax.get_xticks()
    xlabels = [str(float(v / 5e5)) for v in xlocs]
    xlabels[0] = 0
    ax.set_ylim([-0.05, 0.3 ])

    ax.set_xticklabels(xlabels)
    ax.ticklabel_format(fontweight='bold')
    ax.tick_params(labelsize=font_size)
    plt.legend(loc=4,fontsize=font_size)
    plt.gcf().subplots_adjust(bottom=0.15, top=0.97, right=0.975)
    plt.figtext(0.90, 0.03, "1e5", fontsize=font_size, transform=plt.gcf().transFigure)
    plt.ylabel('Average Gain',labelpad=-2, fontsize=font_size, fontweight='bold')
    plt.xlabel('Training Episodes',fontsize=font_size, fontweight='bold')
    plt.show()
    



if __name__ == '__main__':
    main()
