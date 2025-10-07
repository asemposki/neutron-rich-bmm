# BAND SDK v0.2 Community Policy Compatibility for `neutron-rich-bmm`


> This document summarizes the efforts of current and future BAND member packages to achieve compatibility with the BAND SDK community policies.  Additional details on the BAND SDK are available [here](https://raw.githubusercontent.com/bandframework/bandframework/main/resources/sdkpolicies/bandsdk.md) and should be considered when filling out this form. The most recent copy of this template exists [here](https://raw.githubusercontent.com/bandframework/bandframework/main/resources/sdkpolicies/template.md).
>
> This file should filled out and placed in the directory in the `bandframework` repository representing the software name appended by `bandsdk`.  For example, if you have a software `foo`, the compatibility file should be named `foobandsdk.md` and placed in the directory housing the software in the `bandframework` repository. No open source code can be included without this file.
>
> All code included in this repository will be open source.  If a piece of code does not contain a open-source LICENSE file as mentioned in the requirements below, then it will be automatically licensed as described in the LICENSE file in the root directory of the bandframework repository.
>
> Please provide information on your compatibility status for each mandatory policy and, if possible, also for recommended policies. If you are not compatible, state what is lacking and what are your plans on how to achieve compliance. For current BAND SDK packages: If you were not fully compatible at some point, please describe the steps you undertook to fulfill the policy. This information will be helpful for future BAND member packages.
>
> To suggest changes to these requirements or obtain more information, please contact [BAND](https://bandframework.github.io/team).

<!-- #region -->
**Website:** https://github.com/asemposki/neutron-rich-bmm 
**Contact:** Alexandra C. Semposki (<as727414@ohio.edu>)    
**Icon:** https://github.com/asemposki/neutron-rich-bmm/blob/main/BMM_Logo.png
**Description:** `neutron-rich-bmm` gives the community an application of Gaussian process (GP) based Bayesian model mixing (BMM) strategies to the dense matter equation of state. It is meant to serve as a proof of principal for GP-based BMM, and includes a custom kernel for non-stationary GP inference. 

### Mandatory Policies

**BAND SDK**

| # | Policy                 |Support| Notes                                                                                                                                                        |
|---|-----------------------|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1. | Support BAND community GNU Autoconf, CMake, or other build options. |Full| The source code is written in Python, so it does not need `Cmake` or `Autoconf`                                                                              
| 2. | Have a README file in the top directory that states a specific set of testing procedures for a user to verify the software was installed and run correctly. |Full| None.                                                                                                                                                        |
| 3. | Provide a documented, reliable way to contact the development team. |Full| The point of contact provided is at <as727414@ohio.edu>, and is listed in the `README.md` file.                                                                                    |
| 4. | Come with an open-source license |Full| Uses the GPL-3.0 license.                                                                                                                                        |
| 5. | Provide a runtime API to return the current version number of the software. |None| **Still needs to be provided.**                                                                                                                                                    |
| 6. | Provide a BAND team-accessible repository. |Full| The repository is located at https://github.com/asemposki/neutron-rich-bmm and is publicly available to all members of the community. |
| 7. | Must allow installing, building, and linking against an outside copy of all imported software that is externally developed and maintained .|Full| None.                                                                                                                                                        |
| 8. |  Have no hardwired print or IO statements that cannot be turned off. |Full| None.                                                                                                                                                        |


### Recommended Policies

| # | Policy                 | Support | Notes                                                                                                          |
|---|------------------------|---------|----------------------------------------------------------------------------------------------------------------|
|**R1.**| Have a public repository. | Full    | None.                                                                                                          |
|**R2.**| Free all system resources acquired as soon as they are no longer needed. | Full    | None.                                                                                                          |
|**R3.**| Provide a mechanism to export ordered list of library dependencies. | None    | None.                                                                                                          |
|**R4.**| Document versions of packages that it works with or depends upon, preferably in machine-readable form.  | Partial | The repository contains install instructions for install virtual environments and installing the dependencies. |
|**R5.**| Have SUPPORT, LICENSE, and CHANGELOG files in top directory.  | Partial | Has the LICENSE file in the directory but no SUPPORT or CHANGELOG files.                                                                   |
|**R6.**| Have sufficient documentation to support use and further development.  | Partial | Has docstrings in the source code, and will have full API available soon.                                                                       |
|**R7.**| Be buildable using 64-bit pointers; 32-bit is optional. | Full    | Package supports 64 bit pointers.                                                            |
|**R8.**| Do not assume a full MPI communicator; allow for user-provided MPI communicator. | N/A     | None.                                                                                                          |
|**R9.**| Use a limited and well-defined name space (e.g., symbol, macro, library, include). | Full    | None.                                                                                                          |
|**R10.**| Give best effort at portability to key architectures. | Full    | None.                                                                                                          |
|**R11.**| Install headers and libraries under `<prefix>/include` and `<prefix>/lib`, respectively. | N/A     | None.                                                                                                          |
|**R12.**| All BAND compatibility changes should be sustainable. | Full    | None.                                                                                                          |
|**R13.**| Respect system resources and settings made by other previously called packages. | Full    | None.                                                                                                          |
|**R14.**| Provide a comprehensive test suite for correctness of installation verification. | None | Currently no testing available.                                     |
<!-- #endregion -->
