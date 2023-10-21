# OUTRE
This is the code repository of "OUTRE: An Out-of-core De-Redundancy Framework for GNN Training on Massive Graphs within A Single Server". The code of OUTRE is built on an existing GNN training framework [Ginex](https://github.com/SNU-ARC/Ginex). The Bloom Filter implementation in OUTRE is from [here](https://github.com/ArashPartow/bloom).

#### Setup:

1. Disable `read_ahead` on Linux.
    ```console
    sudo -s
    echo 0 > /sys/block/$block_device_name/queue/read_ahead_kb
    ```

2. Install necessary Linux packages. 
    1. `sudo apt-get install -y build-essential`
    2. `sudo apt-get install -y cgroup-tools`
    3. `sudo apt-get install -y unzip`
    4. `sudo apt-get install -y python3-pip` and `pip3 install --upgrade pip`
    5. Compatible NVIDIA CUDA driver and toolkit.

3. Install Python packages. 
    1. PyTorch
    2. ogb
    3. PyG
    4. DGL with version of >= 1.0
    5. others that necessary

4. Install ninja.

    ```console
    sudo wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
    sudo unzip ninja-linux.zip -d /usr/local/bin/
    sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
    ```

5. Use cgroup to limit the memory size. For example, we limit the host memory size to 64GB.

    ```console
    sudo -s
    cgcreate -g memory:64gb
    echo 64000000000 > /sys/fs/cgroup/memory/64gb/memory.limit_in_bytes
    ```

6. Allocate enough swap area.

#### Run on mag240m-cite:

1. Prepare dataset
    ```console
    python3 prepare_dataset_mag.py --dataset mag240m
    ```

2. Partition the original graph

    ```console
    python3 partition_fennel_twolevel.py --dataset mag240m
    ```

3. Create neighbor cache

    ```console
    python3 create_neigh_cache.py --neigh-cache-size 10000000000
    ````

4. Get `PYTHONPATH`
    ```console
    python3 get_pythonpath.py
    ```

5. Run OUTRE on mag240m-cite. Replace `PYTHONPATH=...` with the outcome of step 4.
    ```console
    sudo PYTHONPATH=xxx cgexec -g memory:64gb python3 -W ignore run_profiling.py --neigh-cache-size 10000000000 --feature-cache-size 30000000000 --dataset mag240m
    
    sudo PYTHONPATH=xxx cgexec -g memory:64gb python3 -W ignore run_main.py --neigh-cache-size 10000000000 --feature-cache-size 30000000000 --num-epochs 1 --dataset mag240m
    ```
