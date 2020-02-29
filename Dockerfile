FROM dandavison7/birdnet
RUN mkdir elaenia
WORKDIR elaenia
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN pip install .
WORKDIR /
