---
search:
  exclude: true
---
# 沙箱客户端

使用本页选择沙箱工作应在哪里运行。在大多数情况下，`SandboxAgent` 定义保持不变，仅需在 [`SandboxRunConfig`][agents.run_config.SandboxRunConfig] 中更改沙箱客户端和客户端特定选项。

!!! warning "Beta 功能"

    沙箱智能体目前处于 Beta 阶段。在正式发布前，API 细节、默认值和支持的功能可能会发生变化，并且预计未来会提供更多高级功能。

## 决策指南

<div class="sandbox-nowrap-first-column-table" markdown="1">

| 目标 | 首选 | 原因 |
| --- | --- | --- |
| 在 macOS 或 Linux 上实现最快的本地迭代 | `UnixLocalSandboxClient` | 无需额外安装，便于使用本地文件系统进行开发。 |
| 基本的容器隔离 | `DockerSandboxClient` | 使用特定镜像在 Docker 内运行工作。 |
| 托管执行或生产环境级隔离 | 托管式沙箱客户端 | 将工作区边界移至由提供商管理的环境。 |

</div>

## 本地客户端

对于大多数用户，建议从以下两种沙箱客户端之一开始：

<div class="sandbox-nowrap-first-column-table" markdown="1">

| 客户端 | 安装 | 适用场景 | 示例 |
| --- | --- | --- | --- |
| `UnixLocalSandboxClient` | 无 | 在 macOS 或 Linux 上实现最快的本地迭代。适合作为本地开发的默认选择。 | [Unix 本地入门示例](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/unix_local_runner.py) |
| `DockerSandboxClient` | `openai-agents[docker]` | 需要容器隔离，或需要使用特定镜像在本地复现目标环境。 | [Docker 入门示例](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/docker/docker_runner.py) |

</div>

Unix-local 是基于本地文件系统进行开发的最简便方式。当需要更强的环境隔离或与生产环境保持一致时，请改用 Docker 或托管提供商。

`SandboxPathGrant.host_path` 仅适用于 Docker，用于将主机路径映射到容器内的另一个 POSIX 路径。Unix-local 仅支持同路径授权。详情请参阅[清单路径授权](guide.md#manifest)。

要从 Unix-local 切换到 Docker，请保持智能体定义不变，仅更改运行配置：

```python
from docker import from_env as docker_from_env

from agents.run import RunConfig
from agents.sandbox import SandboxRunConfig
from agents.sandbox.sandboxes.docker import DockerSandboxClient, DockerSandboxClientOptions

run_config = RunConfig(
    sandbox=SandboxRunConfig(
        client=DockerSandboxClient(docker_from_env()),
        options=DockerSandboxClientOptions(image="python:3.14-slim"),
    ),
)
```

当需要容器隔离，或希望沙箱镜像与其他环境中使用的镜像保持一致时，请使用此方式。请参阅 [examples/sandbox/docker/docker_runner.py](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/docker/docker_runner.py)。

## 挂载与远程存储

挂载条目描述要公开的存储；挂载策略描述沙箱后端如何连接该存储。可从 `agents.sandbox.entries` 导入内置挂载条目和通用策略。托管提供商策略可从 `agents.extensions.sandbox` 或特定于提供商的扩展包中获取。

常用挂载选项：

- `mount_path`：存储在沙箱中显示的位置。相对路径基于清单根目录解析；绝对路径则按原样使用。
- `read_only`：默认为 `True`。仅当沙箱应将更改写回已挂载存储时，才设置为 `False`。
- `mount_strategy`：必填。所用策略必须同时匹配挂载条目和沙箱后端。

挂载会被视为临时工作区条目。快照和持久化流程会分离或跳过已挂载路径，而不会将已挂载的远程存储复制到保存的工作区中。

通用本地/容器策略：

<div class="sandbox-nowrap-first-column-table" markdown="1">

| 策略或模式 | 适用场景 | 说明 |
| --- | --- | --- |
| `InContainerMountStrategy(pattern=RcloneMountPattern(...))` | 沙箱镜像可以运行 `rclone`。 | 支持 S3、GCS、R2、Azure Blob 和 Box。`RcloneMountPattern` 可在 `fuse` 模式或 `nfs` 模式下运行。 |
| `InContainerMountStrategy(pattern=MountpointMountPattern(...))` | 镜像中包含 `mount-s3`，并且需要 Mountpoint 风格的 S3 或 S3 兼容访问。 | 支持 `S3Mount` 和 `GCSMount`。 |
| `InContainerMountStrategy(pattern=FuseMountPattern(...))` | 镜像中包含 `blobfuse2` 并支持 FUSE。 | 支持 `AzureBlobMount`。 |
| `InContainerMountStrategy(pattern=S3FilesMountPattern(...))` | 镜像中包含 `mount.s3files`，并且可以访问现有的 S3 Files 挂载目标。 | 支持 `S3FilesMount`。 |
| `DockerVolumeMountStrategy(driver=...)` | Docker 应在容器启动前连接由卷驱动程序支持的挂载。 | 仅适用于 Docker。S3、GCS、R2、Azure Blob 和 Box 可通过 `rclone` 挂载；S3 和 GCS 也可通过 `mountpoint` 挂载。 |

</div>

## 支持的托管平台

当需要托管环境时，通常可以沿用同一份 `SandboxAgent` 定义，仅需在 [`SandboxRunConfig`][agents.run_config.SandboxRunConfig] 中更改沙箱客户端。

如果使用的是已发布的 SDK，而非此仓库的检出版本，请通过匹配的软件包 extra 安装沙箱客户端依赖项。

有关特定于提供商的设置说明以及仓库中扩展代码示例的链接，请参阅 [examples/sandbox/extensions/README.md](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/README.md)。

<div class="sandbox-nowrap-first-column-table" markdown="1">

| 客户端 | 安装 | 示例 |
| --- | --- | --- |
| `BlaxelSandboxClient` | `openai-agents[blaxel]` | [Blaxel 运行器](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/blaxel_runner.py) |
| `CloudflareSandboxClient` | `openai-agents[cloudflare]` | [Cloudflare 运行器](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/cloudflare_runner.py) |
| `DaytonaSandboxClient` | `openai-agents[daytona]` | [Daytona 运行器](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/daytona/daytona_runner.py) |
| `E2BSandboxClient` | `openai-agents[e2b]` | [E2B 运行器](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/e2b_runner.py) |
| `ModalSandboxClient` | `openai-agents[modal]` | [Modal 运行器](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/modal_runner.py) |
| `RunloopSandboxClient` | `openai-agents[runloop]` | [Runloop 运行器](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/runloop/runner.py) |
| `VercelSandboxClient` | `openai-agents[vercel]` | [Vercel 运行器](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/vercel_runner.py) |

</div>

托管式沙箱客户端会公开特定于提供商的挂载策略。请选择最适合所用存储提供商的后端和挂载策略：

<div class="sandbox-nowrap-first-column-table" markdown="1">

| 后端 | 挂载说明 |
| --- | --- |
| Docker | 支持将 `S3Mount`、`GCSMount`、`R2Mount`、`AzureBlobMount`、`BoxMount` 和 `S3FilesMount` 与 `InContainerMountStrategy`、`DockerVolumeMountStrategy` 等本地策略配合使用。 |
| `ModalSandboxClient` | 支持使用 `ModalCloudBucketMountStrategy` 以及 `S3Mount`、`R2Mount` 和通过 HMAC 身份验证的 `GCSMount` 挂载云存储桶。可以使用内联凭据或已命名的 Modal Secret。 |
| `CloudflareSandboxClient` | 支持使用 `CloudflareBucketMountStrategy` 以及 `S3Mount`、`R2Mount` 和通过 HMAC 身份验证的 `GCSMount` 挂载存储桶。 |
| `BlaxelSandboxClient` | 支持将 `BlaxelCloudBucketMountStrategy` 与 `S3Mount`、`R2Mount` 或 `GCSMount` 条目配对，以挂载云存储桶。还支持通过 `BlaxelDriveMount` 和 `BlaxelDriveMountStrategy` 使用持久化 Blaxel Drives，二者均可从 `agents.extensions.sandbox.blaxel` 获取。 |
| `DaytonaSandboxClient` | 支持使用 `DaytonaCloudBucketMountStrategy` 通过 `rclone` 挂载云存储；可将其与 `S3Mount`、`GCSMount`、`R2Mount`、`AzureBlobMount` 和 `BoxMount` 配合使用。 |
| `E2BSandboxClient` | 支持使用 `E2BCloudBucketMountStrategy` 通过 `rclone` 挂载云存储；可将其与 `S3Mount`、`GCSMount`、`R2Mount`、`AzureBlobMount` 和 `BoxMount` 配合使用。 |
| `RunloopSandboxClient` | 支持使用 `RunloopCloudBucketMountStrategy` 通过 `rclone` 挂载云存储；可将其与 `S3Mount`、`GCSMount`、`R2Mount`、`AzureBlobMount` 和 `BoxMount` 配合使用。 |
| `VercelSandboxClient` | 支持将 `VercelCloudBucketMountStrategy` 与 `S3Mount` 条目配对，以挂载仅能在创建时配置的 S3 和 S3 兼容存储桶；已挂载的会话无法恢复，并且内联凭据需要 `allow_s3_credential_exposure=True`。 |

</div>

下表总结了每种后端可直接挂载哪些远程存储条目。

<div class="sandbox-nowrap-first-column-table" markdown="1">

| 后端 | AWS S3 | Cloudflare R2 | GCS | Azure Blob Storage | Box | S3 Files |
| --- | --- | --- | --- | --- | --- | --- |
| Docker | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ModalSandboxClient` | ✓ | ✓ | ✓ | - | - | - |
| `CloudflareSandboxClient` | ✓ | ✓ | ✓ | - | - | - |
| `BlaxelSandboxClient` | ✓ | ✓ | ✓ | - | - | - |
| `DaytonaSandboxClient` | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `E2BSandboxClient` | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `RunloopSandboxClient` | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `VercelSandboxClient` | ✓ | - | - | - | - | - |

</div>

如需更多可运行的代码示例，请浏览 [examples/sandbox/](https://github.com/openai/openai-agents-python/tree/main/examples/sandbox)，其中包含本地运行、编码、记忆、任务转移和智能体组合模式；有关托管式沙箱客户端，请浏览 [examples/sandbox/extensions/](https://github.com/openai/openai-agents-python/tree/main/examples/sandbox/extensions)。