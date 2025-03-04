# Contribute to this project

Thank you for your interest in this project! 

## How to contribute?

### 1. Report a problem 
If you find a bug or have a suggestion for improvement:
- First check if a similar issue already exists.
- If not, open an [issue](https://github.com/Ophelie-Renaud/vis-generator/issues) clearly describing the problem.

### 2. Suggest a change 
If you want to make a change:
1. **Clone** the repository locally.
	```bash
	# clone ssh (recommanded)
	git clone git@gitlab-research.centralesupelec.fr:dark-era/simsdp-generic-imaging-pipeline.git
	# or clone https
	git clone https://gitlab-research.centralesupelec.fr/dark-era/simsdp-generic-imaging-pipeline.git
	cd vis-generator 
	```

2. **Create a descriptive** branch (`feature-new-feature` or `fix-correction-bug` for example).
    ```bash
    # create a branch
    git checkout -b feature-new-feature
    #add modification
    git add.
    # create a commit
    git commit -m "Message"
    # send your work to Github
    git push origin feature-new-feature
    ```

3. **Submit a pull request** explaining your changes.
    - Open a [Pull request](https://github.com/Ophelie-Renaud/vis-generator/pulls).
    - Click on `New pull request`.
    - Add a description of your changes.
    - Click on `Create pull request`.

---

## On going features

- [ ] Automating pipeline computation timing fitting function
  - [x] on CPU modeling
  - [ ] on GPU modeling
- [ ] Simulating on fixed parameter
  - [x] all frequency channels on 1 architecture node
  - [x] one frequency channel per architecture node
- [x] Automating moldable parameter simulation browsing
- [ ] Executing pipeline on generated MeasurementSet and compare output quality with configured *true sky*.
