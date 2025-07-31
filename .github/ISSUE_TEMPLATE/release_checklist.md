---
name: Release checklist
about: 'Maintainers only: Checklist for making a new release'
title: 'Release vX.Y.Z'
labels: 'maintenance'
assignees: ''

---


<!-- Optional -->
**Target date:** YYYY/MM/DD

## Update dependencies

- [ ] Check all the dependency changes since the last release are  reflected in `environment.yml`
- [ ] Update the backup environment file `env/environment.yml` with `make conda_export`
- [ ] update PolarToolkit version in `environment.yml` in [PolarToolkit-Binder repo](https://github.com/mdtanker/polartoolkit-binder) to the latest version number

## Update the changelog

- [ ] Generate a list of commits between the last release tag and now: `git log HEAD...v1.2.3 > changes.md`
- [ ] Use this to summarize the major changes to the code and add short descriptions to `CHANGELOG.md` (not `docs/changelog.md`!).
- [ ] Add contributors to the list
- [ ] Open a PR to update the changelog
- [ ] Merge the PR

## Make a release

After the changelog PR is merged:

- [ ] Draft a new release on GitHub
- [ ] The tag and release name should be a version number (following Semantic Versioning) with a leading `v` (`v1.5.7`)
- [ ] Fill the release description with a Markdown version of the latest changelog entry
- [ ] Publish the release

## Conda-forge package

A PR should be opened automatically on the project feedstock repository.

- [ ] Add/remove/update any dependencies that have changed in `meta.yaml`
- [ ] If dropping/adding support for Python/numpy versions, make sure the correct version restrictions are applied in `meta.yaml`
- [ ] Merge the PR
