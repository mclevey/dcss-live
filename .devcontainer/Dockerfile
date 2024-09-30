FROM python:3.11-bullseye

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install --no-install-recommends \
    apt-utils \
    dialog \
    git \
    iproute2 \
    procps \
    lsb-release \
    curl \
    tree \
    wget \
    unzip \
    graphviz \
    graphviz-dev \
    xdg-utils \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install Quarto
RUN wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.3.450/quarto-1.3.450-linux-amd64.tar.gz \
    && tar -xvzf quarto-1.3.450-linux-amd64.tar.gz \
    && cp -r quarto-1.3.450 /opt/quarto \
    && ln -s /opt/quarto/bin/quarto /usr/local/bin/quarto \
    && rm -rf quarto-1.3.450-linux-amd64.tar.gz quarto-1.3.450

# Create a non-root user
RUN useradd -m -s /bin/bash vscode

USER vscode
WORKDIR /home/vscode

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Install Starship (for current user only)
RUN curl -sS https://starship.rs/install.sh | sh -s -- --yes --bin-dir=$HOME/.local/bin \
    && echo 'eval "$(starship init bash)"' >> ~/.bashrc

ENV PATH="/home/vscode/.local/bin:$PATH"

# Install TinyTeX (as vscode user)
RUN quarto install tinytex

# Set up the workspace
WORKDIR /workspace
COPY --chown=vscode:vscode ../pyproject.toml ../poetry.lock ../README.md ./
COPY --chown=vscode:vscode ../dcss ./dcss

RUN poetry install \
    && poetry run pip install -e . \
    && poetry run python -m spacy download en_core_web_sm

CMD ["/bin/bash"]