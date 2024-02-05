# run from /flush5/tay400/

import seaborn as sns
import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--csv_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--plot_title", type=str)
    parser.add_argument("--widen", type=bool, default=False)

    args, _ = parser.parse_known_args()

    dataframe = pd.read_csv(args.csv_file)
    dataframe = pd.DataFrame(data=dataframe)
    # print(dataframe)

    dataframe_melted = pd.melt(
        dataframe, 
        id_vars="method", 
        value_vars=list(dataframe.keys())[2:-1],
        value_name="dice score",
        var_name="region"
    )
    # print(dataframe_melted)

    if args.widen: 
        plt.figure(figsize=(60, 30))
        plt.xticks(rotation=30, ha='right', fontsize=22)
        plt.yticks(fontsize=34)
        params = {
            'legend.fontsize': 30,
            'legend.title_fontsize': 30
        }
        plt.rcParams.update(params)
    
    plt.ylim(0, 1)
    plot = sns.boxplot(
        data=dataframe_melted, 
        x="region", 
        y="dice score", 
        hue="method"
    )

    if args.widen:
        plot.axes.set_title(args.plot_title, fontsize=50)
        plot.set_xlabel("region", fontsize=30)
        plot.set_ylabel("dice score", fontsize=30)
    else:
        plot.axes.set_title(args.plot_title)

    fig = plot.get_figure()

    fig.savefig(args.output_dir)

    print(f"box plot saved at {args.output_dir}.")


if __name__ == "__main__":
    main()
