---
search:
  exclude: true
---
# 샌드박스 클라이언트

이 페이지에서 샌드박스 작업을 실행할 위치를 선택합니다. 대부분의 경우 `SandboxAgent` 정의는 그대로 유지하고 [`SandboxRunConfig`][agents.run_config.SandboxRunConfig]에서 샌드박스 클라이언트와 클라이언트별 옵션만 변경합니다.

!!! warning "베타 기능"

    샌드박스 에이전트는 베타 버전입니다. 정식 출시 전까지 API 세부 사항, 기본값, 지원 기능이 변경될 수 있으며, 시간이 지나면서 더 많은 고급 기능이 추가될 예정입니다.

## 결정 가이드

<div class="sandbox-nowrap-first-column-table" markdown="1">

| 목표 | 시작할 항목 | 이유 |
| --- | --- | --- |
| macOS 또는 Linux에서 가장 빠른 로컬 반복 개발 | `UnixLocalSandboxClient` | 추가 설치 없이 간단한 로컬 파일 시스템에서 개발할 수 있습니다. |
| 기본적인 컨테이너 격리 | `DockerSandboxClient` | 특정 이미지를 사용하는 Docker 내부에서 작업을 실행합니다. |
| 호스티드 실행 또는 프로덕션 수준의 격리 | 호스티드 샌드박스 클라이언트 | 작업 공간 경계를 공급자가 관리하는 환경으로 이동합니다. |

</div>

## 로컬 클라이언트

대부분의 사용자는 다음 두 샌드박스 클라이언트 중 하나로 시작하는 것이 좋습니다.

<div class="sandbox-nowrap-first-column-table" markdown="1">

| 클라이언트 | 설치 | 선택할 상황 | 예제 |
| --- | --- | --- | --- |
| `UnixLocalSandboxClient` | 없음 | macOS 또는 Linux에서 가장 빠르게 로컬 반복 개발을 진행하려는 경우. 로컬 개발에 적합한 기본 선택지입니다. | [Unix-local 시작 예제](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/unix_local_runner.py) |
| `DockerSandboxClient` | `openai-agents[docker]` | 컨테이너 격리가 필요하거나 대상 환경을 로컬에서 재현하기 위해 특정 이미지를 사용하려는 경우. | [Docker 시작 예제](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/docker/docker_runner.py) |

</div>

Unix-local은 로컬 파일 시스템을 대상으로 개발을 시작하는 가장 쉬운 방법입니다. 더 강력한 환경 격리 또는 프로덕션 수준의 환경 일치가 필요하면 Docker나 호스티드 공급자로 전환합니다.

`SandboxPathGrant.host_path`은 Docker 전용이며 호스트 경로를 컨테이너 내부의 다른 POSIX 경로에 매핑합니다. Unix-local은 동일 경로 권한 부여만 지원합니다. 자세한 내용은 [매니페스트 경로 권한 부여](guide.md#manifest)를 참조하세요.

Unix-local에서 Docker로 전환하려면 에이전트 정의는 그대로 유지하고 실행 구성만 변경합니다.

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

컨테이너 격리가 필요하거나 샌드박스 이미지가 다른 환경에서 사용하는 이미지와 일치해야 할 때 이 방법을 사용합니다. [examples/sandbox/docker/docker_runner.py](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/docker/docker_runner.py)를 참조하세요.

## 마운트 및 원격 스토리지

마운트 항목은 노출할 스토리지를 나타내고, 마운트 전략은 샌드박스 백엔드가 해당 스토리지를 연결하는 방식을 나타냅니다. 기본 제공 마운트 항목과 범용 전략은 `agents.sandbox.entries`에서 가져옵니다. 호스티드 공급자용 전략은 `agents.extensions.sandbox` 또는 공급자별 확장 패키지에서 사용할 수 있습니다.

일반적인 마운트 옵션은 다음과 같습니다.

- `mount_path`: 샌드박스에서 스토리지가 나타나는 위치입니다. 상대 경로는 매니페스트 루트를 기준으로 해석되며, 절대 경로는 그대로 사용됩니다.
- `read_only`: 기본값은 `True`입니다. 샌드박스가 마운트된 스토리지에 변경 사항을 다시 기록해야 하는 경우에만 `False`을 설정합니다.
- `mount_strategy`: 필수입니다. 마운트 항목과 샌드박스 백엔드 모두에 맞는 전략을 사용합니다.

마운트는 임시 작업 공간 항목으로 처리됩니다. 스냅샷 및 영속성 처리 과정에서는 마운트된 원격 스토리지를 저장된 작업 공간에 복사하지 않고 마운트된 경로를 분리하거나 건너뜁니다.

범용 로컬/컨테이너 전략은 다음과 같습니다.

<div class="sandbox-nowrap-first-column-table" markdown="1">

| 전략 또는 패턴 | 사용할 상황 | 참고 |
| --- | --- | --- |
| `InContainerMountStrategy(pattern=RcloneMountPattern(...))` | 샌드박스 이미지에서 `rclone`을 실행할 수 있는 경우. | S3, GCS, R2, Azure Blob 및 Box를 지원합니다. `RcloneMountPattern`은 `fuse` 모드 또는 `nfs` 모드로 실행할 수 있습니다. |
| `InContainerMountStrategy(pattern=MountpointMountPattern(...))` | 이미지에 `mount-s3`이 있고 Mountpoint 방식의 S3 또는 S3 호환 액세스가 필요한 경우. | `S3Mount` 및 `GCSMount`을 지원합니다. |
| `InContainerMountStrategy(pattern=FuseMountPattern(...))` | 이미지에 `blobfuse2` 및 FUSE 지원이 있는 경우. | `AzureBlobMount`을 지원합니다. |
| `InContainerMountStrategy(pattern=S3FilesMountPattern(...))` | 이미지에 `mount.s3files`이 있고 기존 S3 Files 마운트 대상에 연결할 수 있는 경우. | `S3FilesMount`를 지원합니다. |
| `DockerVolumeMountStrategy(driver=...)` | 컨테이너가 시작되기 전에 Docker가 볼륨 드라이버 기반 마운트를 연결해야 하는 경우. | Docker 전용입니다. S3, GCS, R2, Azure Blob 및 Box는 `rclone`을 통해 마운트할 수 있으며, S3와 GCS는 `mountpoint`를 통해서도 마운트할 수 있습니다. |

</div>

## 지원되는 호스티드 플랫폼

호스티드 환경이 필요한 경우 일반적으로 동일한 `SandboxAgent` 정의를 그대로 사용하고 [`SandboxRunConfig`][agents.run_config.SandboxRunConfig]에서 샌드박스 클라이언트만 변경합니다.

이 저장소의 체크아웃 대신 배포된 SDK를 사용하는 경우 일치하는 패키지 extra를 통해 샌드박스 클라이언트 종속성을 설치합니다.

저장소에 포함된 확장 코드 예제의 공급자별 설정 참고 사항과 링크는 [examples/sandbox/extensions/README.md](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/README.md)를 참조하세요.

<div class="sandbox-nowrap-first-column-table" markdown="1">

| 클라이언트 | 설치 | 예제 |
| --- | --- | --- |
| `BlaxelSandboxClient` | `openai-agents[blaxel]` | [Blaxel 실행 예제](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/blaxel_runner.py) |
| `CloudflareSandboxClient` | `openai-agents[cloudflare]` | [Cloudflare 실행 예제](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/cloudflare_runner.py) |
| `DaytonaSandboxClient` | `openai-agents[daytona]` | [Daytona 실행 예제](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/daytona/daytona_runner.py) |
| `E2BSandboxClient` | `openai-agents[e2b]` | [E2B 실행 예제](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/e2b_runner.py) |
| `ModalSandboxClient` | `openai-agents[modal]` | [Modal 실행 예제](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/modal_runner.py) |
| `RunloopSandboxClient` | `openai-agents[runloop]` | [Runloop 실행 예제](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/runloop/runner.py) |
| `VercelSandboxClient` | `openai-agents[vercel]` | [Vercel 실행 예제](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/vercel_runner.py) |

</div>

호스티드 샌드박스 클라이언트는 공급자별 마운트 전략을 제공합니다. 스토리지 공급자에 가장 적합한 백엔드와 마운트 전략을 선택합니다.

<div class="sandbox-nowrap-first-column-table" markdown="1">

| 백엔드 | 마운트 참고 사항 |
| --- | --- |
| Docker | `InContainerMountStrategy` 및 `DockerVolumeMountStrategy` 같은 로컬 전략을 사용하여 `S3Mount`, `GCSMount`, `R2Mount`, `AzureBlobMount`, `BoxMount`, `S3FilesMount`를 지원합니다. |
| `ModalSandboxClient` | `S3Mount`, `R2Mount` 및 HMAC 인증 방식의 `GCSMount`과 함께 `ModalCloudBucketMountStrategy`을 사용하여 클라우드 버킷 마운트를 지원합니다. 인라인 자격 증명 또는 이름이 지정된 Modal Secret을 사용할 수 있습니다. |
| `CloudflareSandboxClient` | `S3Mount`, `R2Mount` 및 HMAC 인증 방식의 `GCSMount`과 함께 `CloudflareBucketMountStrategy`을 사용하여 버킷 마운트를 지원합니다. |
| `BlaxelSandboxClient` | `BlaxelCloudBucketMountStrategy`을 `S3Mount`, `R2Mount` 또는 `GCSMount` 항목과 함께 사용하여 클라우드 버킷 마운트를 지원합니다. 또한 `BlaxelDriveMount` 및 `BlaxelDriveMountStrategy`을 사용하여 영구 Blaxel Drives를 지원하며, 둘 다 `agents.extensions.sandbox.blaxel`에서 사용할 수 있습니다. |
| `DaytonaSandboxClient` | `DaytonaCloudBucketMountStrategy`을 사용해 `rclone`을 통한 클라우드 스토리지 마운트를 지원합니다. `S3Mount`, `GCSMount`, `R2Mount`, `AzureBlobMount`, `BoxMount`와 함께 사용합니다. |
| `E2BSandboxClient` | `E2BCloudBucketMountStrategy`를 사용해 `rclone`를 통한 클라우드 스토리지 마운트를 지원합니다. `S3Mount`, `GCSMount`, `R2Mount`, `AzureBlobMount`, `BoxMount`과 함께 사용합니다. |
| `RunloopSandboxClient` | `RunloopCloudBucketMountStrategy`을 사용해 `rclone`를 통한 클라우드 스토리지 마운트를 지원합니다. `S3Mount`, `GCSMount`, `R2Mount`, `AzureBlobMount`, `BoxMount`과 함께 사용합니다. |
| `VercelSandboxClient` | `VercelCloudBucketMountStrategy`을 `S3Mount` 항목과 함께 사용하여 생성 시점에만 S3 및 S3 호환 버킷 마운트를 지원합니다. 마운트된 세션은 재개할 수 없으며, 인라인 자격 증명을 사용하려면 `allow_s3_credential_exposure=True`가 필요합니다. |

</div>

아래 표에는 각 백엔드가 직접 마운트할 수 있는 원격 스토리지 항목이 요약되어 있습니다.

<div class="sandbox-nowrap-first-column-table" markdown="1">

| 백엔드 | AWS S3 | Cloudflare R2 | GCS | Azure Blob Storage | Box | S3 Files |
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

실행 가능한 코드 예제를 더 살펴보려면 로컬, 코딩, 메모리, 핸드오프 및 에이전트 구성 패턴은 [examples/sandbox/](https://github.com/openai/openai-agents-python/tree/main/examples/sandbox)에서, 호스티드 샌드박스 클라이언트는 [examples/sandbox/extensions/](https://github.com/openai/openai-agents-python/tree/main/examples/sandbox/extensions)에서 확인하세요.