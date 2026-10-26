"""Linux 실행 의존성을 프로젝트 직접 구현 소스와 구분하는지 검사합니다."""
import unittest
from unittest.mock import patch

from tools import validate_artifacts


class DependencyScopeTests(unittest.TestCase):
    def test_installed_venv_is_not_walked_as_product_source(self):
        root = validate_artifacts.ROOT
        observed = []

        def walk(path):
            directories = ['.venv', 'node_modules', 'web']
            yield str(root), directories, []
            observed.extend(directories)
            # 제외가 실패하면 외부 패키지 경로의 잘못된 소스에 접근해 실패합니다.
            if '.venv' in directories:
                yield str(root / '.venv'), [], ['foreign_dependency.py']

        with patch.object(validate_artifacts.os, 'walk', side_effect=walk):
            result = validate_artifacts.validate(root)
        self.assertNotIn('.venv', observed)
        self.assertNotIn('node_modules', observed)
        self.assertIn('web', observed)
        self.assertEqual(result['errors'], [])
