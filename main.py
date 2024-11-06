import click
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def standardize_data(_df: pd.DataFrame) -> pd.DataFrame:
    df = _df.copy()
    # Standardize the data
    scaler = StandardScaler()
    array = scaler.fit_transform(df)
    return pd.DataFrame(array, columns=df.columns, index=df.index)


def fit_pca(_df: pd.DataFrame, exclude_preweights=list):
    df = _df.copy()

    # もしもexclude_preweightsが空でない場合、その行を削除する
    if exclude_preweights:
        df = df.drop(exclude_preweights, axis=0)

    # Perform PCA
    pca = PCA()
    pca.fit(df)
    return pca


def transform_pca(_df: pd.DataFrame, pca: PCA, exclude_transform=list):
    df = _df.copy()

    # もしもexclude_transformが空でない場合、その行を削除する
    if exclude_transform:
        df = df.drop(exclude_transform, axis=0)

    # Transform the data
    array = pca.transform(df)
    return pd.DataFrame(
        array,
        columns=[f"PC{i}" for i in range(1, array.shape[1] + 1)],
        index=df.index,
    )


def scatter_pca(df: pd.DataFrame, axes: list, output: str):
    if len(axes) != 2:
        raise ValueError("axes must be a list of length 2")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(f"PC{axes[0]}")
    ax.set_ylabel(f"PC{axes[1]}")
    ax.set_title("PCA")

    # 行名ごとに色を変えてプロット
    for name in df.index.unique():
        ax.scatter(
            df.loc[name, f"PC{axes[0]}"],
            df.loc[name, f"PC{axes[1]}"],
            label=name,
            alpha=0.5,
        )
    ax.legend()
    # gridを表示
    ax.grid()
    plt.tight_layout()
    fig.savefig(output)


def plot_weights(data: pd.Series, output_path: str):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xlabel("variable")
    ax.set_ylabel("weights")

    ax.bar(data.index, data.to_numpy(), )
    ax.grid(axis="y")
    for label in ax.get_xticklabels():
        label.set_rotation(90)

    ax = fig.add_subplot(2, 1, 2)
    ax.plot((data - data.shift(periods=1)).abs())
    ax.grid()
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    ax.set_xlabel("variable")
    ax.set_ylabel("$|w_{i} - w_{i-1}|$")
    plt.tight_layout()
    fig.savefig(output_path)


@click.group()
@click.option(
    "--standardize",
    "-std",
    required=False,
    is_flag=True,
    default=False,
    help="Standardize the data",
)
@click.option(
    "--exclude-preweights",
    "-ep",
    multiple=True,
    help="Exclude preweight index",
)
@click.option(
    "--exclude-transform",
    "-et",
    multiple=True,
    help="Exclude transform index",
)
@click.pass_context
def cmd(ctx, standardize, exclude_preweights, exclude_transform):
    ctx.ensure_object(dict)
    ctx.obj["standardize"] = standardize
    ctx.obj["exclude_preweights"] = exclude_preweights
    ctx.obj["exclude_transform"] = exclude_transform


@cmd.command()
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def run(ctx, path):
    options = ctx.obj
    standardize = options["standardize"]
    exclude_preweights = options["exclude_preweights"]
    exclude_transform = options["exclude_transform"]

    df = pd.read_csv(path, header=0, index_col=0)
    click.echo("> Raw data")
    click.echo(df)
    # click.echo(df.index.unique())

    if standardize:
        df = standardize_data(df)

    pca = fit_pca(df, list(exclude_preweights))
    transformed = transform_pca(df, pca, list(exclude_transform))
    click.echo("> Transformed to PCA")
    click.echo(transformed)

    return transformed, pca


@cmd.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--axis", "-ax", type=int, default=1, help="Display PC{N}'s weight")
@click.option("--num", "-n", type=int, default=10, help="Display the top N weights")
@click.option("--output", "-o", type=click.Path(), default="img/weights.png")
@click.pass_context
def weights(ctx, path, axis, num, output):

    df = pd.read_csv(path, header=0, index_col=0)
    transformed, pca = ctx.invoke(run, path=path)

    # 各変数の重み（固有ベクトル）を表示
    weights = pd.DataFrame(
        pca.components_,
        columns=df.columns,
        index=[f"PC{i}" for i in range(1, len(transformed.columns) + 1)],
    )

    # PC{axis}の重みを絶対値にして降順に表示
    click.echo(f"> PC{axis}'s weight, top {num}")
    click.echo(weights.loc[f"PC{axis}"].abs().sort_values(ascending=False).head(num))
    plot_weights(weights.loc[f"PC{axis}"].abs().sort_values(ascending=False).head(num), output)


@cmd.command()
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def explained(ctx, path):
    transformed, pca = ctx.invoke(run, path=path)
    explained_variance_ratio = pca.explained_variance_ratio_
    for i, ratio in enumerate(explained_variance_ratio):
        click.echo(f"PC{i + 1}:\t{ratio:.2f}")
        if ratio < 0.01:
            click.echo(f"Since the contribution rate of PC{i + 1} is less than 1%, subsequent components will be ignored.")
            break


@cmd.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="img/scatter.png")
@click.option("--pca-axes", "-ax", type=int, multiple=True, default=[1, 2])
@click.pass_context
def scatter(ctx, path, output, pca_axes):
    transformed, _ = ctx.invoke(run, path=path)

    if len(pca_axes) != 2:
        raise ValueError("scatter-axes must be a list of length 2")
    scatter_pca(transformed, pca_axes, output)


def main():
    cmd()


if __name__ == '__main__':
    main()
