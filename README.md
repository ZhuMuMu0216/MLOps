# This repository is for the project of course: [DTU-MLOps](https://skaftenicki.github.io/dtu_mlops/#course-organization)

## Project Description: Deep Learning in Classification - Hotdog/Not Hotdog

The primary goal of this project is to train a deep learning model that classifies images as either containing a hotdog or not. It's not a difficult task, but it's a good way for us to get familiar with the Machine Learning Operations.

<!-- This task, while simple in its premise, serves as a practical application of image classification using deep learning and demonstrates how neural networks can learn to distinguish between similar and dissimilar objects. Such classification tasks have wide-ranging implications, from food recognition in mobile applications to broader fields like medical imaging and autonomous driving. -->

For this project, we will leverage the dataset available on Kaggle: [Hotdog-NotHotdog](https://www.kaggle.com/datasets/thedatasith/hotdog-nothotdog). This dataset provides labeled images of hotdogs and non-hotdogs,and it has already been split into training set and test set, which makes it an ideal candidate for training and evaluating a supervised deep learning model. While the dataset’s size and diversity will influence the model’s performance, basic preprocessing steps, such as resizing images to a fixed size (e.g., 128x128), normalizing pixel values, and data augmentation (e.g., rotations, flips), will be applied to increase model robustness. We will utilize transfer learning by initializing ResNet18 with pre-trained weights on ImageNet, fine-tuning it for the hotdog classification task.

1. The deep learning framework chosen for this project is [PyTorch](https://github.com/huggingface/pytorch-image-models), known for its flexibility and ease of use in implementing neural networks. PyTorch provides comprehensive libraries and tools for training, validation, and visualization, making it the ideal choice for this image classification task. 

2. The neural network architecture we will employ is [ResNet](https://arxiv.org/pdf/1512.03385) which is proposed by Kaiming in 2016, known for its effectiveness in extracting features from images due to its residual connections, which alleviate the vanishing gradient problem in deep networks.

During the whole project, we will focuses on getting organized and be familiar with good development practices. Besides, we aim to have good version control and ensure the reproducibility. We below tools to achieve the whole project.

1. We use a [cookiecutter template](https://github.com/SkafteNicki/mlops_template) for getting started with Machine Learning Operations (MLOps).

2. Use the [GitHub](https://github.com/) (to store our code) and [DVC](https://dvc.org/) (to control data version, we usually store our data in Google Drive or GCP-cloud storage) to achieve the version control.

3. We use [Docker](https://skaftenicki.github.io/dtu_mlops/s3_reproducibility/docker/) and [Config Files](https://skaftenicki.github.io/dtu_mlops/s3_reproducibility/config_files/) to make our project reproducible.

4. During developing the code, we use [pdb](https://docs.python.org/3/library/pdb.html) to debug, use [loguru](https://loguru.readthedocs.io/en/stable/) to make logging and use [wandb](https://wandb.ai/site/) to tracking our training results and hardware performance.

5. We use [Unit testing](https://skaftenicki.github.io/dtu_mlops/s5_continuous_integration/unittesting/) to test individual parts of your code base to test for correctness. Besides, we use the [GitHub actions](https://skaftenicki.github.io/dtu_mlops/s5_continuous_integration/github_actions/) to automate the testing, such that it is done every time we push to our repository. If we combine this with only pushing to branches and then only merging these branches whenever all automated testing has passed, our code should be fairly safe against bugs.

6. We use the [GCP](https://cloud.google.com/storage?utm_source=google&utm_medium=cpc&utm_campaign=emea-dk-all-en-dr-bkws-all-all-trial-b-gcp-1707574&utm_content=text-ad-none-any-dev_c-cre_677656980141-adgp_Hybrid+%7C+BKWS+-+MIX+%7C+Txt+-+Storage+-+Cloud+Storage-kwid_43700078358185205-kwd-298160887431-userloc_1005023&utm_term=kw_cloud+google+storage-net_g-plac_&&gad_source=1&gclid=CjwKCAiAhP67BhAVEiwA2E_9g7MwNmFWBQitjl6x7d70GodgOTlA5IIRxzQz1P-SJ_g2eSfNHLzFmhoCvzAQAvD_BwE&gclsrc=aw.ds&hl=en) to build our remote virtual machine, storage our dataset, make CI/CD and use the Vertex AI to 
train our model. It's really a powerful platform made by Google.

7. Deployment(Haven't learned this yet)

8. Monitoring(Haven't learned this yet)


## Start our project!

### 1. Download the overall project files
1. Create your directory in your local machine.

2. Folk our repository / Create the git file and clone our repository

    ```dash
    git init
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
    docker run -v $(pwd)/data:/data -it my_image

    '''In Windows Shell'''
    docker run -v ${PWD}/data:/data -it my_image

5. We can run the docker image right now.
    ```bash
    docker run -v ${PWD}/data:/data -it train_image 