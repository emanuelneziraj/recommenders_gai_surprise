# Hybrid Movie Recommendation System

This repository contains the source code and documentation for a Bachelor's thesis project that explores two different approaches to developing a movie recommendation system: one using KNNBasic, SVD and NMF with the Surprise library for collaborative filtering, and another using OpenAI's GPT models gpt-3.5-turbo and gpt-4-turbo for generating personalized movie recommendations.

## Project Overview

The goal of this project is to compare the effectiveness of traditional machine learning techniques and cutting-edge language models in providing personalized movie recommendations. It utilizes the [MovieLens](https://grouplens.org/datasets/movielens/) latest small dataset to model user preferences and provide recommendations.

## System Requirements

- Python 3.10
- Pandas
- NumPy
- Surprise
- OpenAI's Python API

## Clone

```Bash
git clone https://github.com/emanuelneziraj/recommenders_gai_surprise.git
```

## Recommender-GAI

### Configuration

To run this project, you need to configure an OpenAI API Key in a `config.ini` file as followed:

```
[DEFAULT]
GPT_TOKEN = <API_KEY>
```

### Installation

To set up the project, follow these steps:

```bash
cd Recommender-GAI
pip install -r requirements.txt
python main.py
```

### Process

Since GPT is generating recommendations for 600+ users, this will take a long time. Approximately 1.5+ hours per model.

### Output


## Recommenders-Surprise


### Configuration

No Configuration needed.

### Installation

To set up the project, follow these steps:

```bash
cd Recommender-Surprise
pip install -r requirements.txt
python main.py
```

### Output

## Contributions

Contributions to this project are welcome! If you find a bug or have a suggestion for improvement, please create an issue or a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact

For more information, please contact [Emanuel Neziraj](mailto:emanuel.neziraj@edu.fh-joanneum.com).
