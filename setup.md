## Setting up the repository

### Creating GitHub token

- Go to github -> settings -> developer settings -> personal access tokens -> tokens (classic) -> generate new token
- When creating a new token, select all the checkboxes, then generate and copy the token
- Open google drive, create the following folder structure: `AI/project`
- Add git_token.json inside the `project` folder
- Contents of json file:

```
{
    "token":"<your_token>"
}
```
- Then refer `git.ipynb` till **"Clone the repository section"**

## Using github with colab

- Use `git.ipynb` to `add`, `commit`, and `push` changes to the GitHub.


*NOTE:*
- Make sure you are in the right folder. You can check this by running `!ls`
- Use `%cd /content/drive/MyDrive/AI/project/CSGY-6613` to change directory to local repository folder
- You will need to run repository setup just once(the first time)
- Remember to run `!git pull` when restarting work, this will reduce chances for merge conflicts

## Dataset generation
Dataset generation code for sort-of-CLEVR is added. Use this code to generate the dataset as it cannot be uploaded to github because of size restrictions.
