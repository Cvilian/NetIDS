import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.ticker import FormatStrFormatter
from urllib3.exceptions import NewConnectionError, MaxRetryError

def labeling(host_name):
    api_key = ''    # sign up to Virustotal, then acquire your own API key!
    url = 'https://www.virustotal.com/vtapi/v2/url/report'

    while True :
        try :
            params = {'apikey' : api_key,'resource' : host}
            response = requests.get(url, params=params)

            if response.status_code == 200 :
                D = response.json()
                K = D.keys()
                time.sleep(0.5)

                if 'positives' in K and int(D['positives']) > 0:
                    return "Y"
                else :
                    return "N"
                break

            time.sleep(10)
        except (TimeoutError, NewConnectionError, MaxRetryError, ConnectionError):
            time.sleep(10)

def visualize_performance(res):
    fig=plt.figure(figsize=(10, 4.5))

    ax1 = fig.add_subplot(121)
    plt.plot(list(range(0, 101, 10)),
            [res['t_tprs'][max(0, i-1)] for i in range(0, 101, 10)],
            label='$training$', marker='o', markersize=8, linewidth=3, clip_on=False)
    plt.plot(list(range(0, 101, 10)), 
            [res['v_tprs'][max(0, i-1)] for i in range(0, 101, 10)],
            label='$validation$', marker='s', markersize=8, linewidth=3, clip_on=False)

    ax1.set_xlim([0, 100])
    ax1.set_ylim([0.8, 1.0])
    ax1.set_xticks([0, 20, 40, 60, 80, 100])
    ax1.set_yticks([0.8, 0.85, 0.9, 0.95, 1.00])
    ax1.set_xticks(range(0, 101, 10), minor=True)
    ax1.set_yticks([i/40 for i in range(32, 41)], minor=True)
    ax1.set_xlabel("Epochs", fontsize=20)
    ax1.set_ylabel("True Positive Rate", fontsize=20)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax1.legend(markerscale=1, fontsize=16, loc='lower right')
    ax1.grid(which='both', color='#BDBDBD', linestyle='--', linewidth=1)
    plt.rc('font',family='Times New Roman', size=16)

    ax2 = fig.add_subplot(122)
    plt.plot(list(range(0, 101, 10)),
            [res['t_fprs'][max(0, i-1)] for i in range(0, 101, 10)],
            label='$training$', marker='o', markersize=8, linewidth=3, clip_on=False)
    plt.plot(list(range(0, 101, 10)), 
            [res['v_fprs'][max(0, i-1)] for i in range(0, 101, 10)],
            label='$validation$', marker='s', markersize=8, linewidth=3, clip_on=False)

    ax2.set_xlim([0, 100])
    ax2.set_ylim([0.0, 0.1])
    ax2.set_xticks([0, 20, 40, 60, 80, 100])
    ax2.set_yticks([0, 0.025, 0.05, 0.075, 0.1])
    ax2.set_xticks(range(0, 101, 10), minor=True)
    ax2.set_yticks([i/80 for i in range(0, 9)], minor=True)
    ax2.set_xlabel("Epochs", fontsize=20)
    ax2.set_ylabel("True Positive Rate", fontsize=20)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.legend(markerscale=1, fontsize=16)
    ax2.grid(which='both', color='#BDBDBD', linestyle='--', linewidth=1)
    plt.rc('font',family='Times New Roman', size=16)

    fig.tight_layout()
    plt.show()

def visualize_tsne(x, y):
    n = x.shape[0]
    samples = np.random.choice(range(n), n//100, replace=False)

    x_t = x[samples]
    y_t = y[samples]
    n = x_t.shape[0]

    model = TSNE(learning_rate=500, n_components=2, random_state=0)
    transformed = model.fit_transform(x_t)

    for v, label in zip(range(2), ['normal', 'malicious']):
        idx = [i for i in range(n) if y_t[i] == v]
        plt.scatter(transformed[idx, 0], transformed[idx, 1], label=label)

    plt.legend()
    plt.show()
