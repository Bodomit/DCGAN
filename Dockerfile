FROM tensorflow/tensorflow:2.0.0-gpu-py3

# Make the directories in root.
RUN mkdir /src

# Copy requiremenets file to the src dir.
WORKDIR /src
COPY requirements.txt /src

# Install pip modules
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code.
COPY . /src