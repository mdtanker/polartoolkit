# PolarToolkit

```{include} ../README.md
:start-after: <!-- SPHINX-START-proj-desc -->
:end-before: <!-- SPHINX-END-proj-desc -->
```

```{include} ../README.md
:start-after: <!-- SPHINX-START-badges -->
:end-before: <!-- SPHINX-END-badges -->
```

```{image} cover_fig.png
:alt: cover figure
:width: 400px
:align: center
```

```{admonition} Ready for daily use but still changing.
:class: seealso

This means that we are still adding a lot of new features and sometimes we make changes to the ones we already have while we try to improve the software based on users' experience, test new ideas, make better design decisions, etc. Some of these changes could be **backwards incompatible**. Keep that in mind before you update polartoolkit to a new major version (i.e. from `v1.0.0` to `v2.0.0`).
```


```{include} ../README.md
:start-after: <!-- SPHINX-START-long-desc -->
:end-before: <!-- SPHINX-END-long-desc -->
```


```{admonition} How to contribute
:class: seealso

I really welcome all forms of contribution! If you have any questions, comments or suggestions, please open a [discussion](https://github.com/mdtanker/polartoolkit/discussions/new/choose) or [issue (feature request)](https://github.com/mdtanker/polartoolkit/issues/new/choose) on the [GitHub page](https://github.com/mdtanker/polartoolkit/)!

Also, please feel free to share how you're using PolarToolkit, I'd love to know.

Please, read our [Contributor Guide](https://polartoolkit.readthedocs.io/en/latest/contributing.html) to learn
how you can contribute to the project.
```

```{note}
*Many parts of this documentation was adapted from the* [Fatiando project](https://www.fatiando.org/).
```

```{toctree}
:maxdepth: 1
:hidden:
overview
install
gallery/index.md
tutorial/index.md
```

```{toctree}
:hidden:
:caption: 📂 Available datasets
datasets/index.md
datasets/antarctica/index.md
datasets/greenland_arctic/index.md
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Tips
tips.ipynb
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: 📖 Reference documentation
api/polartoolkit
citing.md
changelog.md
references.rst
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: ℹ️ Other resources
contributing.md
Source code on GitHub <https://github.com/mdtanker/polartoolkit>
```


::::{grid} 2
:::{grid-item-card} {octicon}`rocket` Getting started?
:text-align: center
New to PolarToolkit? Start here!
```{button-ref} overview
    :click-parent:
    :color: primary
    :outline:
    :expand:
```
:::

:::{grid-item-card} {octicon}`comment-discussion` Need help?
:text-align: center
Start a discussion on GitHub!
```{button-link} https://github.com/mdtanker/polartoolkit/discussions
    :click-parent:
    :color: primary
    :outline:
    :expand:
    Discussions
```
:::

:::{grid-item-card} {octicon}`file-badge` Reference documentation
:text-align: center
A list of modules and functions
```{button-ref} api/polartoolkit
    :click-parent:
    :color: primary
    :outline:
    :expand:
```
:::

:::{grid-item-card} {octicon}`bookmark` Using PolarToolkit for research?
:text-align: center
Citations help support our work
```{button-ref} citing
    :click-parent:
    :color: primary
    :outline:
    :expand:
```
:::
::::