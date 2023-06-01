import json
import matplotlib.pyplot as plt
import numpy as np
markers={'forward_push':'^',
             'power_iteration':'o',
             'heavy_ball':'<',
             'nag':'x',
             'power_push':'s',
             'forward_push_sor':'v',
             'power_push_sor':'h'
             }
indexes=['A','B','C','D','E','F']
colors={'forward_push':'tab:gray',
             'power_iteration':'tab:blue',
             'heavy_ball':'tab:brown',
             'nag':'tab:purple',
             'power_push':'tab:green',
             'forward_push_sor':'tab:orange',
             'power_push_sor':'tab:red'}
names={'forward_push':'FwdPush',
        "power_iteration":'PowItr',
        "heavy_ball":'HB',
        "nag":'NAG',
        "power_push":'PwrPush',
        "power_push_sor":'PwrPushSOR',
        "forward_push_sor":'FwdPushSOR'}
dataset_names={
        'webs':'web-Stanford',
        'lj':'livejournal',
        'pokec':'pokec',
        'dblp':'dblp',
        'products':'products',
        'orkut':'orkut'}
        
datasets=['dblp','orkut','products','webs','lj','pokec']
#setting alpha
alpha=0.2
fig, axs = plt.subplots(1, 6, figsize=(36, 5))

methods=dict()
for i,dataset in enumerate(datasets):
    if dataset in ['dblp','orkut','products']:
        methods=[
                    'forward_push'    ,
                    'power_iteration'   ,
                    'heavy_ball'        ,
                    'nag'               ,
                    'power_push'        ,
                    'power_push_sor'    ,
                    'forward_push_sor'   
                    ]
    elif dataset in ['lj','pokec','webs']:
        methods=['forward_push'          ,
                    'power_iteration'   ,
                    'power_push'        ,
                    'forward_push_sor'  ,
                    'power_push_sor'    ]
    results={}

    with open(f'result/{dataset}_{alpha}.json','r') as f:
        raw=f.readlines()
    for j in raw:
        cur_point=json.loads(j)
        cur_method=list(cur_point.keys())[0]
        cur_result=results.get(cur_method,[])
        cur_result=cur_result+cur_point[cur_method]
        results[cur_method]=cur_result

    method_result={}
    for method in methods:
        points=results[method]
        time_list=np.array([k[1] for k in points],dtype=float)
        l1_error=np.array([k[2] for k in points],dtype=float)
        x,y,z=[0],[8],[0]
        inters=[1,5e-1,1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5,5e-6,1e-6,5e-7,1e-7,5e-8,1e-8,5e-9]
        for j in range(len(inters)-1):
            index=(l1_error<inters[j]) *  (l1_error>=inters[j+1])
            x.append(time_list[index].mean())
            y.append(np.log10(l1_error[index].mean()*1e8))
        axs[i].plot(x,y,marker=markers[method],
                markersize=8.,
                markerfacecolor='white', markeredgewidth=.5,
                color=colors[method], label=method,
                linewidth=.5,fillstyle='none')

        axs[i].set_yticks([0,2,4,6,8],[1,r'$10^2$',r'$10^4$',r'$10^6$',r'$10^8$'],fontsize=15)
        axs[i].set_title(r'$\ell_1$-error ($\times 10^{-8}$)',loc='left',fontsize=15)
        axs[i].set_xlabel(r'time',fontsize=15)
        axs[i].set_title(f'({indexes[i]}) {dataset}',y=-0.24,fontsize=20,
            fontfamily="Times New Roman"
            )
    axs[i].grid()

axs[0].legend(loc='center right', fontsize=12,
                    bbox_to_anchor=(3, 1.2),ncols=7,)
fig.savefig(f'figs/time_{alpha}.pdf', dpi=600, bbox_inches='tight', pad_inches=0,format='pdf')

#Oper
fig, axs = plt.subplots(1, 6, figsize=(36, 5))
for i,dataset in enumerate(datasets):
    if dataset in ['dblp','orkut','products']:
        methods=[
                    'forward_push'    ,
                    'power_iteration'   ,
                    'heavy_ball'        ,
                    'nag'               ,
                    'power_push'        ,
                    'power_push_sor'    ,
                    'forward_push_sor'   
                    ]
    elif dataset in ['lj','pokec','webs']:
        methods=['forward_push'          ,
                    'power_iteration'   ,
                    'power_push'        ,
                    'forward_push_sor'  ,
                    'power_push_sor'    ]
    with open(f'result/{dataset}_{alpha}.json','r') as f:
        raw=f.readlines()
    for j in raw:
        cur_point=json.loads(j)
        cur_method=list(cur_point.keys())[0]
        cur_result=results.get(cur_method,[])
        cur_result=cur_result+cur_point[cur_method]
        results[cur_method]=cur_result
    
    for method in methods:
        points=results[method]
        time_list=np.array([k[0] for k in points],dtype=float)
        l1_error=np.array([k[2] for k in points],dtype=float)
        x,y=[0],[8]
        inters=[1,5e-1,1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5,5e-6,1e-6,5e-7,1e-7,5e-8,1e-8,5e-9]
        for j in range(len(inters)-1):
            index=(l1_error<inters[j]) *  (l1_error>=inters[j+1])
            x.append(time_list[index].mean())
            y.append(np.log10(l1_error[index].mean()*1e8))
        axs[i].plot(x,y,marker=markers[method],
                markersize=8.,
                markerfacecolor='white', markeredgewidth=.5,
                color=colors[method], label=method,
                linewidth=.5,fillstyle='none')

        axs[i].set_yticks([0,2,4,6,8],[1,r'$10^2$',r'$10^4$',r'$10^6$',r'$10^8$'],fontsize=15)
        axs[i].set_title(r'$\ell_1$-error ($\times 10^{-8}$)',loc='left',fontsize=15)
        axs[i].set_xlabel(r'#updates',fontsize=15)
        axs[i].set_title(f'({indexes[i]}) {dataset}',y=-0.24,fontsize=20,
            fontfamily="Times New Roman"
            )
    axs[i].grid()
fig.savefig(f'figs/oper_{alpha}.pdf', dpi=600, bbox_inches='tight', pad_inches=0,format='pdf')