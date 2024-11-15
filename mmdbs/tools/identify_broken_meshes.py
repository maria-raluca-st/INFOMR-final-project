import pandas as pd
import multiprocessing
import subprocess
from tqdm import tqdm


def run_script(path):
    try:
        result = subprocess.run(
            ["python", "norm_vert_script.py", path],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return (path, result.returncode)
    except subprocess.CalledProcessError as e:
        return (path, e.returncode)
    except subprocess.TimeoutExpired as e:
        return (path, 124)


def main():
    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    df = pd.read_csv("INFOMR-final-project/mmdbs/data/metadata.csv")

    # Get the list of paths from the DataFrame
    paths = df["Path"].tolist()

    # Run the script for each path in parallel
    results = list(tqdm(pool.imap(run_script, paths), total=len(paths)))
    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    results_df = pd.DataFrame(results, columns=["Path", "ReturnCode"])

    # Save the results to a CSV file
    results_df.to_csv("shape_manifest.csv", index=False)

    print(results_df["ReturnCode"].value_counts())


if __name__ == "__main__":
    main()
