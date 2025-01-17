from locust import HttpUser, between, task


class MyUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    wait_time = between(3, 5)

    @task
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the FastAPI app."""
        self.client.get("/")

    @task(3)
    def post_predict(self) -> None:
        """A task that simulates a user uploads an image file and invoke the inference endpoint."""
        # We hardcoded the image file since it won't really affect the test result
        with open("data\\test\\nothotdog\\food (1).jpg", "rb") as file:
            self.client.post("/predict", files={"data": file})
