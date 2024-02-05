import pandas as pd
from argparse import ArgumentParser
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt

def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--dataset_csv", type=str)
    parser.add_argument("--plot_csv", type=str)

    args, _ = parser.parse_known_args()

    dataset_csv = pd.read_csv(args.dataset_csv)
    plot_csv = pd.read_csv(args.plot_csv)
    ratings = []

    # loop through rows in plot_csv and assign difficulty rating in new column
    for i in range(len(plot_csv)):
        info = plot_csv['info'][i]
        info_split = info.split('_')

        moving_id = f'{info_split[1]}_{info_split[2]}'
        fixed_id = f'{info_split[4]}_{info_split[5]}'

        moving_index = list(dataset_csv['id']).index(moving_id)
        fixed_index = list(dataset_csv['id']).index(fixed_id)

        moving_rating = dataset_csv['severity'][moving_index]
        fixed_rating = dataset_csv['severity'][fixed_index]

        final_rating = f'{moving_rating}_to_{fixed_rating}'
        ratings.append(final_rating)

    plot_csv['rating'] = ratings
    plot_csv.to_csv(args.plot_csv[:-4] + '_final.csv')

    plot_csv['rating'] = [s.replace('2_to_1', '1_to_2') for s in plot_csv['rating']] 
    plot_csv['rating'] = [s.replace('3_to_1', '1_to_3') for s in plot_csv['rating']]
    plot_csv['rating'] = [s.replace('3_to_2', '2_to_3') for s in plot_csv['rating']]

    # print(pd.Series(plot_csv['rating']).drop_duplicates().tolist())

    # create separate plot for each region
    # for region in df.region.unique():  
    #   sns.boxplot(df[df.region == region], x=severity, y=DICE, hue=method)
    regions = list(plot_csv.keys()[2:-2])
    
    plot_csv_melted = pd.melt(
        plot_csv,
        id_vars=['method', 'rating', 'info'],
        value_vars=regions,
        value_name='dice score',
        var_name='region'
    )
    plot_csv_melted.to_csv(args.plot_csv[:-4] + '_melted.csv')

    # count number of pairs for each rating/method
    # unique_ratings = pd.Series(plot_csv['rating']).drop_duplicates().tolist()
    # methods = pd.Series(plot_csv['method']).drop_duplicates().tolist()

    # for r in unique_ratings:
    #     data = plot_csv[plot_csv['rating'] == r]
    #     print("============================")
    #     print("Rating: ", r)

    #     for m in methods:
    #         print(m, ':', len(list(data['method'] == m)))
    #         # print(data[data['method'] == m])

    OUTPUT_DIR = 'job_data/plots/generalisation_mitii/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for region in regions:
        output_dir = OUTPUT_DIR + f'boxplot_{region}.png'

        # indices = list(np.where(plot_csv_melted['region'] == region)[0])
        # data = plot_csv_melted[indices[0]: indices[-1]]

        data = plot_csv_melted[plot_csv_melted['region'] == region]

        plt.figure()
        plot = sns.boxplot(
            data=data,
            x='rating',
            y='dice score',
            hue='method'
        )
        plot.axes.set_title(f'MITII {region}')

        fig = plot.get_figure()
        fig.savefig(output_dir)
        # print(f'{region} boxplot saved')

    # generalisation gap
    plot_csv_melted = plot_csv_melted.set_index(['info', 'region', 'rating'])
    generalisation_gap = plot_csv_melted.loc[plot_csv_melted['method'] == 'pre-trained on OASIS', 'dice score'] \
        - plot_csv_melted.loc[plot_csv_melted['method'] == 'no extra training', 'dice score']
    # generalisation_gap = pd.DataFrame({'dice score': generalisation_gap})
    generalisation_gap = generalisation_gap.reset_index()

    # print(generalisation_gap)
    # print(generalisation_gap.columns)

    plt.figure()
    plot = sns.boxplot(
        data=generalisation_gap[generalisation_gap['region'].isin(['CSF', 'cortical GM', 'white matter'])],
        x='region',
        y='dice score',
        hue='rating'
    )
    plot.axes.set_title(f'MITII Generalisation')

    fig = plot.get_figure()
    fig.savefig(OUTPUT_DIR + 'generalisation.png')

    print('All plots saved in ' + OUTPUT_DIR)



if __name__ == "__main__":
    main()