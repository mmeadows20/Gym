# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of Compose volume planning.

Short and long volume forms, bind sources under the task folder, refused options, link
resolution and the plan the adapter receives, with the resolver run over the planned document.
"""

import copy
import shutil
from pathlib import Path

import pytest
import yaml

from nemo_gym.sandbox.compose_config import (
    adapt_non_root_sidecars,
    document_volume_sources,
    plan_compose_volumes,
    resolve_compose,
)
from nemo_gym.sandbox.compose_volumes import ComposeVolumeError, parse_volume
from tests.unit_tests.compose_testing import IMAGE_CONFIGS, MAIN, OVERLAY


VOLUME_FIXTURES = Path(__file__).parent / "fixtures" / "harbor_compose_volumes"


class TestComposeVolumes:
    """M22: bind mounts of task files become uploads, named volumes stay for shared storage, external ones map
    through ``volume_sources``; anything else fails naming the field."""

    @pytest.mark.parametrize(
        "entry,expected",
        [
            (
                "./seed.json:/app/seed.json:ro",
                {"type": "bind", "source": "./seed.json", "target": "/app/seed.json", "read_only": True},
            ),
            (
                "/var/lib/models:/models",
                {"type": "bind", "source": "/var/lib/models", "target": "/models", "read_only": False},
            ),
            ("~/cache:/cache:rw,z", {"type": "bind", "source": "~/cache", "target": "/cache", "read_only": False}),
            (
                "pgdata:/var/lib/postgresql/data",
                {"type": "volume", "source": "pgdata", "target": "/var/lib/postgresql/data", "read_only": False},
            ),
            (
                "pgdata:/data:nocopy",
                {
                    "type": "volume",
                    "source": "pgdata",
                    "target": "/data",
                    "read_only": False,
                    "volume": {"nocopy": True},
                },
            ),
            (
                {
                    "type": "bind",
                    "source": "./data",
                    "target": "/data",
                    "read_only": True,
                    "bind": {"create_host_path": True},
                },
                {"type": "bind", "source": "./data", "target": "/data", "read_only": True},
            ),
            (
                {"source": "pgdata", "target": "/data", "volume": {"nocopy": True}, "consistency": "cached"},
                {
                    "type": "volume",
                    "source": "pgdata",
                    "target": "/data",
                    "read_only": False,
                    "volume": {"nocopy": True},
                },
            ),
        ],
    )
    def test_parse_volume_handles_short_and_long_syntax(self, entry, expected):
        assert parse_volume(entry, "api") == expected

    @pytest.mark.parametrize(
        "entry,name",
        [
            ("/data", "/data"),  # anonymous volume
            (":/data", ":/data"),
            ("./a:/data:ro:extra", "./a:/data:ro:extra"),
            ("./a:/data:rshared", "rshared"),
            ({"type": "tmpfs", "target": "/tmp"}, "type"),
            ({"type": "volume", "target": "/data"}, "source"),
            ({"type": "bind", "target": "/data"}, "source"),
            ({"type": "bind", "source": "./a", "target": "data"}, "target"),
            ({"type": "bind", "source": "./a", "target": "/data", "bind": {"propagation": "rshared"}}, "propagation"),
            ({"type": "volume", "source": "v", "target": "/data", "volume": {"subpath": "x"}}, "subpath"),
            ({"type": "volume", "source": "v", "target": "/data", "tmpfs": {}}, "tmpfs"),
            (42, "volumes"),
        ],
    )
    def test_unsupported_volume_entries_name_the_field(self, entry, name):
        with pytest.raises(ComposeVolumeError) as exc:
            parse_volume(entry, "api")
        assert exc.value.name == name

    def test_task_aw_bind_mount_becomes_an_upload_into_the_sidecar(self, tmp_path):
        env = tmp_path / "task-aw" / "environment"
        shutil.copytree(VOLUME_FIXTURES / "task-aw" / "environment", env)
        document = yaml.safe_load((env / "docker-compose.yaml").read_text())
        uploads = plan_compose_volumes(document, env, root=env.parent)
        api = document["services"]["api"]
        assert "volumes" not in api
        assert uploads == {"api": [{"source": str(env / "seed.json"), "target": "/app/seed.json"}]}
        assert api["x-sandbox"] == {"uploads": uploads["api"]}
        assert "x-sandbox" not in document["services"]["main"]
        # Nothing else is needed from the deployment: no shared storage, no volume_sources.
        assert "volumes" not in document

    def test_context_dir_folder_bind_and_task_root_paths_upload(self, tmp_path):
        env = tmp_path / "t" / "environment"
        (env / "data").mkdir(parents=True)
        (env / "data" / "a.txt").write_text("a")
        (tmp_path / "t" / "tests").mkdir()
        document = {
            "services": {
                "main": {"volumes": [f"{env}/data:/data", {"type": "bind", "source": "../tests", "target": "/tests"}]}
            }
        }
        uploads = plan_compose_volumes(document, env, root=env.parent)
        assert uploads["main"] == [
            {"source": str(env / "data"), "target": "/data"},
            {"source": str(tmp_path / "t" / "tests"), "target": "/tests"},
        ]

    def test_named_volumes_stay_in_long_syntax_for_shared_storage(self, tmp_path):
        document = {
            "services": {
                "db": {"volumes": ["pgdata:/var/lib/postgresql/data"]},
                "backup": {
                    "volumes": [{"type": "volume", "source": "pgdata", "target": "/mnt/db", "read_only": True}]
                },
            },
            "volumes": {"pgdata": None},
        }
        assert plan_compose_volumes(document, tmp_path) == {}
        assert document["services"]["db"]["volumes"] == [
            {"type": "volume", "source": "pgdata", "target": "/var/lib/postgresql/data", "read_only": False}
        ]
        assert document["services"]["backup"]["volumes"][0]["read_only"] is True
        # The canonical null declaration is normalised to the mapping the adapter reads (review finding).
        assert document["volumes"] == {"pgdata": {}}

    def test_external_volume_needs_a_volume_sources_entry(self, tmp_path):
        document = {"services": {"db": {"volumes": ["pgdata:/data"]}}, "volumes": {"pgdata": {"external": True}}}
        with pytest.raises(ComposeVolumeError, match="external volume 'pgdata' needs a `volume_sources` entry") as exc:
            plan_compose_volumes(copy.deepcopy(document), tmp_path)
        assert exc.value.name == "pgdata"
        plan_compose_volumes(document, tmp_path, volume_sources={"pgdata": "datasets/pgdata"})
        assert document["services"]["db"]["volumes"][0]["source"] == "pgdata"

    def test_foreign_bind_mount_needs_a_volume_sources_entry(self, tmp_path):
        document = {"services": {"api": {"volumes": ["/var/lib/models:/models:ro"]}}}
        with pytest.raises(ComposeVolumeError, match="outside the task") as exc:
            plan_compose_volumes(copy.deepcopy(document), tmp_path)
        assert exc.value.name == "/var/lib/models"
        plan_compose_volumes(document, tmp_path, volume_sources={"/var/lib/models": "models"})
        assert document["services"]["api"]["volumes"] == [
            {"type": "bind", "source": "/var/lib/models", "target": "/models", "read_only": True}
        ]

    @pytest.mark.parametrize(
        "document,name",
        [
            ({"services": {"db": {"volumes": ["pgdata:/data"]}}}, "pgdata"),
            ({"services": {"db": {"volumes": ["pgdata:/data"]}}, "volumes": {"pgdata": {"driver": "nfs"}}}, "driver"),
            ({"services": {"db": {"volumes": ["./missing.json:/x.json"]}}}, "./missing.json"),
            ({"services": {"db": {"volumes": "./a:/a"}}}, "volumes"),
            ({"services": {"db": {}}, "volumes": []}, "volumes"),
            # Review (round 14): a non-mapping declaration is the planner's own error, naming the volume, not a
            # ``TypeError`` out of ``set(declaration)`` that aborted prepare for every sibling task.
            ({"services": {"db": {"volumes": ["pgdata:/data"]}}, "volumes": {"pgdata": 123}}, "pgdata"),
            ({"services": {"db": {}}, "volumes": {"pgdata": "abc"}}, "pgdata"),
        ],
    )
    def test_undeclared_missing_and_malformed_volumes_fail_naming_them(self, tmp_path, document, name):
        with pytest.raises(ComposeVolumeError) as exc:
            plan_compose_volumes(document, tmp_path)
        assert exc.value.name == name

    def test_a_link_leaving_the_task_folder_is_refused(self, tmp_path):
        env = tmp_path / "t" / "environment"
        env.mkdir(parents=True)
        outside = tmp_path / "outside.json"
        outside.write_text("{}")
        (env / "seed.json").symlink_to(outside)
        with pytest.raises(ComposeVolumeError, match="leaves the task folder") as exc:
            plan_compose_volumes({"services": {"api": {"volumes": ["./seed.json:/seed.json"]}}}, env, root=env.parent)
        assert exc.value.name == "./seed.json"

    def test_a_relative_foreign_bind_resolves_through_volume_sources(self, tmp_path):
        """Review (round 14): ``../../shared:/data`` from ``task/environment`` is ordinary traversal out of the task,
        which Docker binds as a plain host path relative to the Compose file's folder; the symlink-escape check
        compared the unresolved path lexically, took the ``..`` for a link leaving the task, and never reached the
        ``volume_sources`` lookup. It is a foreign source now, like an absolute one, under the literal key."""
        task = tmp_path / "t"
        env = task / "environment"
        env.mkdir(parents=True)
        (tmp_path / "shared").mkdir()
        document = {"services": {"api": {"volumes": ["../../shared:/data"]}}}
        with pytest.raises(ComposeVolumeError, match="outside the task") as exc:
            plan_compose_volumes(copy.deepcopy(document), env, root=task)
        assert exc.value.name == "../../shared"
        assert plan_compose_volumes(document, env, root=task, volume_sources={"../../shared": "shared"}) == {}
        assert document["services"]["api"]["volumes"] == [
            {"type": "bind", "source": "../../shared", "target": "/data", "read_only": False}
        ]
        assert document_volume_sources(document, {"../../shared": "shared", "other": "x"}) == {
            "../../shared": "shared"
        }
        # Traversal that stays inside the task is still an upload, and a link that leaves it is still refused.
        inside = {"services": {"api": {"volumes": ["../environment:/env"]}}}
        assert plan_compose_volumes(inside, env, root=task) == {"api": [{"source": str(env), "target": "/env"}]}
        (tmp_path / "shared" / "x.json").write_text("{}")
        (env / "escape.json").symlink_to(tmp_path / "shared" / "x.json")
        with pytest.raises(ComposeVolumeError, match="leaves the task folder"):
            plan_compose_volumes({"services": {"api": {"volumes": ["./escape.json:/x.json"]}}}, env, root=task)

    def test_a_link_with_an_absolute_target_inside_the_task_is_refused_and_a_relative_one_is_not(self, tmp_path):
        # Review: the server plans on a copy of the task folder (links copied as links). An absolute target still
        # points into the original, so a source reached through one would pass prepare and fail seed; the planner
        # refuses it like the hub refuses such a link in a fetched task. A relative link agrees at both ends.
        task = tmp_path / "t"
        env = task / "environment"
        (env / "realdata" / "sub").mkdir(parents=True)
        (env / "realdata" / "sub" / "a.txt").write_text("a")
        (env / "real.json").write_text("{}")
        (env / "data").symlink_to(env / "realdata", target_is_directory=True)
        (env / "seed.json").symlink_to(env / "real.json")

        def document(*entries):
            return {"services": {"api": {"volumes": list(entries)}}}

        for source, link in [("./data", "data"), ("./data/sub", "data"), ("./seed.json", "seed.json")]:
            with pytest.raises(ComposeVolumeError, match=rf"absolute target: .*environment/{link}; make") as exc:
                plan_compose_volumes(document(f"{source}:/mnt"), env, root=task)
            assert exc.value.name == source
        (env / "data").unlink()
        (env / "seed.json").unlink()
        (env / "data").symlink_to("realdata", target_is_directory=True)
        (env / "seed.json").symlink_to("real.json")
        entries = ("./data:/data", "./data/sub:/sub", "./seed.json:/seed.json")
        uploads = plan_compose_volumes(document(*entries), env, root=task)
        assert uploads == {
            "api": [
                {"source": str(env.resolve() / "realdata"), "target": "/data"},
                {"source": str(env.resolve() / "realdata" / "sub"), "target": "/sub"},
                {"source": str(env.resolve() / "real.json"), "target": "/seed.json"},
            ]
        }
        # The same plan on a copy made the way the server snapshots a task: the links resolve inside the copy.
        copy = tmp_path / "copy"
        shutil.copytree(task, copy, symlinks=True)
        uploads = plan_compose_volumes(document(*entries), copy / "environment", root=copy)
        assert [entry["source"] for entry in uploads["api"]] == [
            str(copy.resolve() / "environment" / "realdata"),
            str(copy.resolve() / "environment" / "realdata" / "sub"),
            str(copy.resolve() / "environment" / "real.json"),
        ]

    def test_a_link_inside_a_bound_folder_is_refused_naming_it(self, tmp_path):
        # Review: a folder is uploaded file by file, so a link inside it would be read through to the host.
        env = tmp_path / "t" / "environment"
        (env / "data" / "sub").mkdir(parents=True)
        (env / "data" / "ok.txt").write_text("ok")
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        (env / "data" / "sub" / "leak").symlink_to(outside)
        with pytest.raises(ComposeVolumeError, match=r"contains a link: .*data/sub/leak") as exc:
            plan_compose_volumes({"services": {"api": {"volumes": ["./data:/data"]}}}, env, root=env.parent)
        assert exc.value.name == "./data"
        # A relative link that stays inside the task is refused too: no link is ever uploaded.
        (env / "data" / "sub" / "leak").unlink()
        (env / "data" / "sub" / "alias").symlink_to("../ok.txt")
        with pytest.raises(ComposeVolumeError, match="contains a link"):
            plan_compose_volumes({"services": {"api": {"volumes": ["./data:/data"]}}}, env, root=env.parent)

    @pytest.mark.parametrize("service", ["main", "api"])
    @pytest.mark.parametrize("source", ["/etc/passwd", "OUTSIDE"])
    def test_a_task_authored_x_sandbox_is_refused_naming_the_field(self, tmp_path, service, source):
        # Review blocker: x-sandbox is the planner's output. A task file carrying one could name any host file
        # as an upload source, so it fails before any volume is looked at, naming the field.
        env = tmp_path / "t" / "environment"
        env.mkdir(parents=True)
        outside = tmp_path / "outside" / "secret"
        outside.parent.mkdir()
        outside.write_text("secret")
        source = str(outside) if source == "OUTSIDE" else source
        document = {
            "services": {
                "main": {},
                "api": {"image": "a"},
            }
        }
        document["services"][service]["x-sandbox"] = {"uploads": [{"source": source, "target": "/tmp/stolen"}]}
        with pytest.raises(ComposeVolumeError, match=f"service {service!r} declares x-sandbox") as exc:
            plan_compose_volumes(document, env, root=env.parent)
        assert exc.value.name == "x-sandbox"
        # Any x-sandbox key, not only uploads: the sidecar adaptation's keys are Gym's too.
        document["services"][service]["x-sandbox"] = {"hosts": []}
        with pytest.raises(ComposeVolumeError) as exc:
            plan_compose_volumes(document, env, root=env.parent)
        assert exc.value.name == "x-sandbox"

    @pytest.mark.parametrize(
        "entry",
        [
            "./seed.json:app/seed.json",
            "./seed.json:/app/../x",
            "data:rel/path",
            {"type": "bind", "source": "./seed.json", "target": "/app/../x"},
            "./seed.json:/",
            {"type": "bind", "source": "./seed.json", "target": "//"},
            "data:/",
        ],
    )
    def test_targets_are_absolute_below_the_root_without_dotdot_in_both_syntaxes(self, tmp_path, entry):
        # Review: the short syntax took any target; a relative or `..` target failed only at seed, and `/` was
        # accepted (chmod and uploads over the whole container).
        (tmp_path / "seed.json").write_text("{}")
        document = {"services": {"api": {"volumes": [entry]}}, "volumes": {"data": {}}}
        with pytest.raises(ComposeVolumeError, match="target") as exc:
            plan_compose_volumes(document, tmp_path)
        assert exc.value.name == "target"

    def test_an_unknown_home_user_in_a_bind_source_names_the_source(self, tmp_path):
        # Review: Path.expanduser raised a bare RuntimeError for `~user` with no such user.
        document = {"services": {"db": {"volumes": ["~nosuchuser_xyz/x:/x"]}}}
        with pytest.raises(ComposeVolumeError, match="cannot resolve the home directory") as exc:
            plan_compose_volumes(document, tmp_path)
        assert exc.value.name == "~nosuchuser_xyz/x"

    def test_logs_tree_binds_on_main_become_adapter_log_paths_and_are_refused_on_a_sidecar(self, tmp_path):
        # Review (round 13): the binds used to be dropped, but in a Compose group nothing created /logs/* before
        # main's command and healthcheck ran (the server's mkdir comes after the group is up), and a bind with a
        # target elsewhere (`/logs/artifacts:/app/output`) lost its target. They now instruct the adapter, which
        # creates the source and links the target before the command, keeping the real target.
        document = {
            "services": {
                "main": {"volumes": ["/logs/verifier:/logs/verifier", "/logs:/logs", "/logs/artifacts:/app/output"]}
            }
        }
        assert plan_compose_volumes(document, tmp_path) == {}
        main = document["services"]["main"]
        assert "volumes" not in main
        assert main["x-sandbox"] == {
            "logs": [
                {"source": "/logs/verifier", "target": "/logs/verifier"},
                {"source": "/logs", "target": "/logs"},
                {"source": "/logs/artifacts", "target": "/app/output"},
            ]
        }
        with pytest.raises(ComposeVolumeError, match="only `main` can mount /logs") as exc:
            plan_compose_volumes({"services": {"api": {"volumes": ["/logs/x:/elsewhere"]}}}, tmp_path)
        assert exc.value.name == "/logs/x"
        # A main that also binds a task file carries both instructions under the one planner key.
        (tmp_path / "environment").mkdir()
        (tmp_path / "environment" / "seed.json").write_text("{}")
        document = {"services": {"main": {"volumes": ["/logs/agent:/logs/agent", "./seed.json:/app/seed.json"]}}}
        uploads = plan_compose_volumes(document, tmp_path / "environment", root=tmp_path)
        assert document["services"]["main"]["x-sandbox"] == {
            "uploads": uploads["main"],
            "logs": [{"source": "/logs/agent", "target": "/logs/agent"}],
        }

    def test_a_bind_source_the_task_digest_skips_is_refused(self, tmp_path):
        # Review (round 13): the planner accepted any bind source under the task root, but content_hash covers
        # only Harbor's packager set (environment/, tests/, solution/, steps/ and the top-level files), so an
        # edit to <task>/data/seed.json changed neither the digest nor the manifest while the server uploaded the
        # edited bytes. The hash stays the publisher's (parity); the planner is the guard.
        task = tmp_path / "t"
        env = task / "environment"
        (env / "data" / "__pycache__").mkdir(parents=True)
        (env / "data" / "__pycache__" / "x.pyc").write_bytes(b"")
        (env / "data" / "a.txt").write_text("a")
        (env / "seed.json").write_text("{}")
        (task / "data").mkdir()
        (task / "data" / "seed.json").write_text("{}")
        (task / "notes.txt").write_text("n")
        (task / "tests").mkdir()
        for source in ("../data/seed.json", "../notes.txt", "./data/__pycache__/x.pyc", ".."):
            document = {"services": {"api": {"volumes": [f"{source}:/x"]}}}
            with pytest.raises(ComposeVolumeError, match="which the task digest does not cover") as exc:
                plan_compose_volumes(document, env, root=task)
            assert exc.value.name == source
        # Covered sources plan as before: a file under environment/, the tests/ folder, the top-level task file,
        # and a covered folder that holds a byte-code cache after a run (the leftover does not change identity).
        document = {
            "services": {
                "api": {"volumes": ["./seed.json:/app/seed.json", "../tests:/tests", "./data:/data"]},
                "main": {"volumes": ["../instruction.md:/instruction.md"]},
            }
        }
        (task / "instruction.md").write_text("do it")
        uploads = plan_compose_volumes(document, env, root=task)
        assert [u["source"] for u in uploads["api"]] == [
            str(env / "seed.json"),
            str(task / "tests"),
            str(env / "data"),
        ]
        assert uploads["main"] == [{"source": str(task / "instruction.md"), "target": "/instruction.md"}]

    def test_a_gpu_reserving_sidecar_resolves_external_volumes_through_the_gpu_mapping(self, tmp_path):
        # Review (round 13): the server routes a GPU-reserving sidecar of a CPU task to the GPU block, so its
        # external volumes resolve through that block's mapping, not the task's; main follows the task.
        gpu_sidecar = {
            "image": "img",
            "deploy": {"resources": {"reservations": {"gpus": 1}}},
            "volumes": ["models:/models"],
        }
        document = {
            "services": {"main": {"image": "img"}, "model": copy.deepcopy(gpu_sidecar)},
            "volumes": {"models": {"external": True}},
        }
        # Mapped only in the GPU block: passes once the GPU mapping is given.
        with pytest.raises(ComposeVolumeError, match="external volume 'models'"):
            plan_compose_volumes(copy.deepcopy(document), tmp_path, volume_sources={})
        plan_compose_volumes(copy.deepcopy(document), tmp_path, volume_sources={}, gpu_volume_sources={"models": "g"})
        # A GPU block without the mapping does not fall back to the task's block.
        with pytest.raises(ComposeVolumeError, match="external volume 'models'"):
            plan_compose_volumes(
                copy.deepcopy(document), tmp_path, volume_sources={"models": "c"}, gpu_volume_sources={}
            )
        # A sidecar that reserves no GPUs, and main, stay on the task's mapping whatever the GPU block maps.
        cpu_document = copy.deepcopy(document)
        del cpu_document["services"]["model"]["deploy"]
        cpu_document["services"]["main"]["volumes"] = ["models:/models"]
        with pytest.raises(ComposeVolumeError, match="external volume 'models'"):
            plan_compose_volumes(
                copy.deepcopy(cpu_document), tmp_path, volume_sources={}, gpu_volume_sources={"models": "g"}
            )
        plan_compose_volumes(cpu_document, tmp_path, volume_sources={"models": "c"}, gpu_volume_sources={})

    def test_document_volume_sources_keeps_only_what_the_document_resolves_externally(self, tmp_path):
        # Review (round 13): the adapter treats any key of its mapping as an externally managed volume, so the
        # deployment-wide mapping must be narrowed to what this document declares external or binds from outside.
        document = {
            "services": {
                "db": {"volumes": ["pgdata:/var/lib/postgresql/data"]},
                "api": {"volumes": ["models:/models", "/var/lib/weights:/weights:ro"]},
                "main": {},
            },
            "volumes": {"pgdata": None, "models": {"external": True}},
        }
        mapping = {
            "pgdata": "datasets/pgdata",  # another task's external volume: this one declares pgdata ordinary
            "models": "shared/models",
            "/var/lib/weights": "weights",
            "unused": "elsewhere",
        }
        plan_compose_volumes(document, tmp_path, volume_sources=mapping)
        assert document_volume_sources(document, mapping) == {"models": "shared/models", "/var/lib/weights": "weights"}
        assert (
            document_volume_sources({}, mapping) == {} and document_volume_sources({"services": None}, mapping) == {}
        )

    def test_documents_without_services_or_volumes_are_left_alone(self, tmp_path):
        for document in ({}, {"services": None}, {"services": {"main": None, "api": {"volumes": None}}}):
            assert plan_compose_volumes(copy.deepcopy(document), tmp_path) == {}

    def test_non_root_sidecar_adaptation_keeps_uploads(self, tmp_path):
        seed = tmp_path / "seed.json"
        seed.write_text("{}")
        overlay = copy.deepcopy(OVERLAY)
        overlay["services"]["browser"]["volumes"] = ["./seed.json:/app/seed.json:ro"]
        uploads = plan_compose_volumes(overlay, tmp_path)
        document = adapt_non_root_sidecars(resolve_compose(overlay, MAIN, IMAGE_CONFIGS), IMAGE_CONFIGS)
        browser = document["services"]["browser"]
        assert browser["x-sandbox"] == {
            "uploads": uploads["browser"],
            "hosts": [],
            "resolve_environment": ["BROWSER_URL"],
        }
