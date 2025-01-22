# **Start our project on your local machine!**

**ATTENTION:** Since we did lots of operations on the GCP(for example, downloading the pre-trained model from cloud storage), you also need to create your own GCP account and replace the project id, bucket name etc.

### 1. Download the overall project files
1. Create your directory in your local machine.

2. Folk our repository / Create the git file and clone our repository

    ```dash
    git clone https://github.com/ZhuMuMu0216/MLOps.git
    ```

### 2. Create venv
1. Create the virtual environment
    ```dash
    python -m venv venv       # create environment

    source venv/bin/activate  # Linux/macOS
    .\venv\Scripts\activate   # Windows

    pip install .             # install all the packages
    ```

### 3. Download the dataset
1. You should enter the `MLOps` directory.

2. Run the below command in your terminal.
    ```
    dvc pull
    ```
### 4. Run the code in docker
1. You should enter the `MLOps` directory.

2. Run the below command in your terminal, and you will build the docker image based on my dockerfile.

    ```bash
    docker build -t train_image -f dockerfiles/train.dockerfile .

3. Check your built docker image.
    ```bash
    docker images
    ```
    You will get below result and we can run the train_image now.

    | REPOSITORY  | TAG    | IMAGE ID      | CREATED        | SIZE  |
    |-------------|--------|---------------|----------------|-------|
    | train_image | latest | 311535037766  | 8 minutes ago  | 6.24GB |

4. As we didn't COPY the data into our docker image, we dynamically mount the Host's `data` Directory to the Container's `/data`.
    ```bash
    '''In Linux system'''
    docker run -v $(pwd)/data:/data -it train_image

    '''In Windows Shell'''
    docker run -v ${PWD}/data:/data -it train_image
    ```

5. Deploy the API
    ```
    docker build -t api-service -f api.dockerfile .
    docker run -d -p 8080:8080 api-service
    ```
    You can access your API at http://localhost:8080

---
