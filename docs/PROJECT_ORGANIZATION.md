# Einstein Bot - Project Organization

## Directory Structure

```
Einstein Bot/
├── proofs/                    # Core proof engine implementation
│   ├── __init__.py           # Package exports
│   ├── theorem_generator.py  # Phase 5A: Theorem generation
│   ├── proof_attempt.py      # Phase 5B: Symbolic proof engine
│   └── utils/                # Proof utilities
│       ├── __init__.py
│       ├── proof_cache.py
│       └── timeout_utils.py
├── validation/               # Phase 3: Validation engine
├── exploration/              # Phase 4: Pattern discovery
├── tests/                    # Comprehensive test suite
│   ├── conftest.py          # Pytest configuration and fixtures
│   ├── test_proof_attempt.py # Phase 5B tests
│   ├── test_theorem_generator.py # Phase 5A tests
│   └── fixtures/            # Test data
├── plans/                    # Implementation planning documents
├── docs/                     # Project documentation
│   ├── summaries/           # Development summaries (not versioned)
│   └── development/         # Development notes (not versioned)
├── results/                  # Generated theorem data
├── main.py                   # CLI interface
└── README.md                # Main project documentation
```

## File Organization Best Practices

### Documentation
- **Public Documentation**: `docs/` - Versioned documentation for users/contributors
- **Development Notes**: `docs/summaries/` and `docs/development/` - Local development notes, not versioned
- **Planning**: `plans/` - Implementation planning documents

### Code Organization
- **Core Logic**: Organized by feature/phase in dedicated packages
- **Tests**: Mirror the source structure with comprehensive test coverage
- **Utilities**: Shared utilities in appropriate sub-packages

### Version Control Exclusions
- Development summaries and notes (personal/temporary documentation)
- Cache directories and temporary files
- Test artifacts and coverage reports
- Virtual environments and IDE configurations

### Testing Strategy
- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component interaction testing
- **Performance Tests**: Caching and execution performance
- **End-to-End Tests**: Complete workflow validation

## Development Workflow

1. **Feature Development**: Create feature branches from main
2. **Testing**: Comprehensive test coverage before commit
3. **Documentation**: Update public docs, keep private notes in docs/summaries/
4. **Code Review**: Follow staff engineer standards
5. **Integration**: Clean, atomic commits with descriptive messages

## Cache and Temporary Files

- `cache/`: Runtime caching (excluded from version control)
- `.pytest_cache/`: Test artifacts (excluded from version control)
- `docs/summaries/`: Development summaries (excluded from version control)
- `docs/development/`: Development notes (excluded from version control) 