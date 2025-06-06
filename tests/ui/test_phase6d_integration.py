import pytest
from unittest.mock import Mock, patch, MagicMock
from ui.app import MathBotUI
from ui.config import UIConfig


class MockSessionState(dict):
    """Mock class for Streamlit session state that behaves like both dict and object."""
    
    def __getattr__(self, item):
        return self.get(item)
    
    def __setattr__(self, key, value):
        self[key] = value


class TestPhase6DIntegration:
    
    @pytest.fixture
    def config(self):
        """Create a mock UIConfig for testing."""
        config = Mock(spec=UIConfig)
        config.ui_settings = {"show_debug": False}
        return config
    
    @pytest.fixture
    def mock_app(self, config):
        """Create a MathBotUI instance with mocked dependencies."""
        with patch('ui.app.get_ui_config', return_value=config), \
             patch('ui.app.TheoremLoader'), \
             patch('ui.app.FormulaLoader'), \
             patch('ui.app.SearchIndex'), \
             patch('ui.app.SearchInterface'), \
             patch('ui.app.TheoremBrowser'), \
             patch('ui.app.TheoremDetail'), \
             patch('ui.app.ProofVisualizer') as mock_proof_viz:
            
            app = MathBotUI()
            app.proof_visualizer = mock_proof_viz.return_value
            return app
    
    @pytest.fixture
    def mock_theorem(self):
        """Create a mock theorem for testing."""
        theorem = Mock()
        theorem.id = "THM_TEST001"
        theorem.short_id = "TEST001"
        theorem.statement = "Test theorem for integration testing"
        theorem.theorem_type_display = "Integration Test"
        
        # Mock validation evidence
        validation_evidence = Mock()
        validation_evidence.pass_rate = 0.95
        validation_evidence.total_tests = 50
        validation_evidence.confidence_level = "High"
        theorem.validation_evidence = validation_evidence
        
        return theorem
    
    @patch('streamlit.set_page_config')
    @patch('streamlit.markdown')
    def test_app_initialization_includes_proof_visualizer(self, mock_markdown, mock_set_page_config, mock_app):
        """Test that the app properly initializes the proof visualizer component."""
        assert hasattr(mock_app, 'proof_visualizer')
        assert mock_app.proof_visualizer is not None
    
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    @patch('streamlit.warning')
    def test_proof_trace_page_rendering_no_theorem(self, mock_warning, mock_markdown, mock_title, mock_app):
        """Test proof trace page renders correctly when no theorem is selected."""
        with patch.object(mock_app, '_get_selected_theorem_for_proof', return_value=None), \
             patch.object(mock_app, '_render_theorem_selection_interface'):
            
            mock_app.render_proof_viewer_page()
            
            # Verify page elements were rendered
            mock_title.assert_called_with("🔍 Proof Trace Visualization")
            mock_warning.assert_called_with("Please select a theorem to view its proof trace.")
    
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    def test_proof_trace_page_rendering_with_theorem(self, mock_markdown, mock_title, mock_app, mock_theorem):
        """Test proof trace page renders correctly with a selected theorem."""
        with patch.object(mock_app, '_get_selected_theorem_for_proof', return_value=mock_theorem), \
             patch('streamlit.session_state', MockSessionState()):
            
            mock_app.render_proof_viewer_page()
            
            # Verify page was rendered
            mock_title.assert_called_with("🔍 Proof Trace Visualization")
            
            # Verify proof visualizer was called
            mock_app.proof_visualizer.render_proof_visualization.assert_called_once()
    
    def test_theorem_selection_integration(self, mock_app, mock_theorem):
        """Test theorem selection for proof visualization integrates with session state."""
        # Test case 1: Explicit proof theorem selection
        with patch('streamlit.session_state', MockSessionState({'selected_theorem_for_proof': mock_theorem})):
            result = mock_app._get_selected_theorem_for_proof()
            assert result == mock_theorem
        
        # Test case 2: Theorem detail selection
        with patch('streamlit.session_state', MockSessionState({'selected_theorem_detail': mock_theorem})):
            result = mock_app._get_selected_theorem_for_proof()
            assert result == mock_theorem
        
        # Test case 3: General theorem selection with ID lookup
        with patch('streamlit.session_state', MockSessionState({'selected_theorem': "THM_TEST001"})):
            # Mock theorem loader
            mock_app.theorem_loader = Mock()
            mock_app.theorem_loader.load_theorems.return_value = [mock_theorem]
            
            result = mock_app._get_selected_theorem_for_proof()
            assert result == mock_theorem
    
    @patch('streamlit.selectbox')
    @patch('streamlit.button')
    @patch('streamlit.markdown')
    def test_theorem_selection_interface(self, mock_markdown, mock_button, mock_selectbox, mock_app, mock_theorem):
        """Test the theorem selection interface renders correctly."""
        mock_selectbox.return_value = 0
        mock_button.return_value = False
        
        # Mock theorem loader
        mock_app.theorem_loader = Mock()
        mock_app.theorem_loader.load_theorems.return_value = [mock_theorem]
        
        mock_app._render_theorem_selection_interface()
        
        # Verify interface elements were rendered
        mock_selectbox.assert_called()
        mock_button.assert_called()
        mock_markdown.assert_called()
    
    def test_navigation_includes_proof_trace(self, mock_app):
        """Test that the navigation sidebar includes the Proof Trace option."""
        # Mock the sidebar components
        mock_sidebar = Mock()
        mock_sidebar.title = Mock()
        mock_sidebar.markdown = Mock()
        mock_sidebar.metric = Mock()
        
        # Create proper context manager mock for expander
        mock_expander = MagicMock()
        mock_expander.__enter__ = Mock(return_value=mock_expander)
        mock_expander.__exit__ = Mock(return_value=None)
        mock_sidebar.expander = Mock(return_value=mock_expander)
        mock_sidebar.selectbox = Mock(return_value="Proof Trace")
        
        with patch('streamlit.sidebar', mock_sidebar), \
             patch('streamlit.selectbox'), \
             patch('streamlit.checkbox'), \
             patch('streamlit.session_state', MockSessionState({'current_page': 'Overview'})), \
             patch('streamlit.rerun'):
            
            mock_app.render_sidebar()
            
            # Verify sidebar selectbox was called with pages including "Proof Trace"
            mock_sidebar.selectbox.assert_called()
            call_args = mock_sidebar.selectbox.call_args
            pages = call_args[0][1]  # Second argument should be the pages list
            assert "Proof Trace" in pages
    
    def test_proof_visualization_session_creation(self, mock_app, mock_theorem):
        """Test that proof visualization sessions are created correctly."""
        with patch.object(mock_app, '_get_selected_theorem_for_proof', return_value=mock_theorem), \
             patch('ui.data.proof_models.ProofVisualizationSession') as mock_session_class, \
             patch('streamlit.session_state', MockSessionState()) as mock_session_state:
            
            mock_app.proof_visualizer.render_proof_visualization = Mock()
            
            mock_app.render_proof_viewer_page()
            
            # Verify session was created
            session_key = f"proof_session_{mock_theorem.id}"
            assert session_key in mock_session_state
    
    def test_proof_visualizer_error_handling(self, mock_app, mock_theorem):
        """Test error handling in proof visualization."""
        with patch.object(mock_app, '_get_selected_theorem_for_proof', return_value=mock_theorem), \
             patch('streamlit.session_state', MockSessionState()), \
             patch('streamlit.error') as mock_error:
            
            # Mock proof visualizer to raise an exception
            mock_app.proof_visualizer.render_proof_visualization = Mock(side_effect=Exception("Test error"))
            
            mock_app.render_proof_viewer_page()
            
            # Verify error was handled
            mock_error.assert_called()
    
    def test_proof_visualizer_not_initialized(self, mock_app):
        """Test handling when proof visualizer is not initialized."""
        mock_app.proof_visualizer = None
        
        with patch('streamlit.error') as mock_error:
            mock_app.render_proof_viewer_page()
            
            mock_error.assert_called_with("Proof visualizer not initialized. Please check the logs.")
    
    def test_page_routing_to_proof_trace(self, mock_app):
        """Test that page routing correctly handles Proof Trace page."""
        with patch('streamlit.session_state', MockSessionState({'current_page': 'Proof Trace'})), \
             patch.object(mock_app, 'render_sidebar'), \
             patch.object(mock_app, 'render_proof_viewer_page') as mock_render_proof:
            
            mock_app.run()
            
            # Verify proof viewer page was called
            mock_render_proof.assert_called_once()
    
    def test_integration_with_phase6a_data_models(self, mock_theorem):
        """Test integration with Phase 6A data models."""
        from ui.data.proof_models import ProofVisualizationSession, ProofTraceData, ProofStep, ProofMethodType
        
        # Test that Phase 6D models work with Phase 6A theorem model
        session = ProofVisualizationSession(
            theorem_id=mock_theorem.id,
            theorem=mock_theorem
        )
        
        assert session.theorem_id == mock_theorem.id
        assert session.theorem == mock_theorem
        
        # Test proof trace data creation
        trace_data = ProofTraceData(theorem_id=mock_theorem.id)
        assert trace_data.theorem_id == mock_theorem.id
    
    def test_integration_with_phase6c_navigation(self, mock_app, mock_theorem):
        """Test integration with Phase 6C search and browse navigation."""
        with patch('streamlit.session_state', MockSessionState({'selected_theorem_detail': mock_theorem})):
            
            result = mock_app._get_selected_theorem_for_proof()
            assert result == mock_theorem
            
            # Test navigation buttons in theorem selection interface
            with patch('streamlit.button', return_value=True) as mock_button, \
                 patch('streamlit.selectbox'), \
                 patch('streamlit.markdown'), \
                 patch('streamlit.rerun') as mock_rerun:
                
                mock_app.theorem_loader = Mock()
                mock_app.theorem_loader.load_theorems.return_value = [mock_theorem]
                
                mock_app._render_theorem_selection_interface()
                
                # Verify navigation buttons were rendered
                assert mock_button.called
    
    def test_proof_service_integration(self, config):
        """Test that proof service integrates correctly with UI config."""
        from ui.services.proof_service import ProofVisualizationService
        
        service = ProofVisualizationService(config)
        assert service.config == config
        assert hasattr(service, 'logger')
    
    def test_mathematical_renderer_integration(self):
        """Test that mathematical renderer integrates correctly."""
        from ui.utils.proof_rendering import MathematicalRenderer
        
        renderer = MathematicalRenderer()
        assert hasattr(renderer, 'logger')
        assert hasattr(renderer, 'render_expression')
        assert hasattr(renderer, 'convert_to_latex')
    
    @patch('streamlit.columns')
    @patch('streamlit.button')
    def test_cross_component_navigation(self, mock_button, mock_columns, mock_app):
        """Test navigation between Phase 6D and other components."""
        # Create context manager mocks for columns
        col1_mock = MagicMock()
        col2_mock = MagicMock()
        col1_mock.__enter__ = Mock(return_value=col1_mock)
        col1_mock.__exit__ = Mock(return_value=None)
        col2_mock.__enter__ = Mock(return_value=col2_mock)
        col2_mock.__exit__ = Mock(return_value=None)
        mock_columns.return_value = [col1_mock, col2_mock]
        mock_button.return_value = True
        
        with patch('streamlit.session_state', MockSessionState()), \
             patch('streamlit.rerun') as mock_rerun, \
             patch('streamlit.selectbox'), \
             patch('streamlit.markdown'), \
             patch('streamlit.warning'):  # Mock warning for no theorems case
            
            # Create a mock theorem to ensure buttons are rendered
            mock_theorem = Mock()
            mock_theorem.short_id = "T001"
            mock_theorem.statement = "Test theorem for navigation"
            
            mock_app.theorem_loader = Mock()
            mock_app.theorem_loader.load_theorems.return_value = [mock_theorem]
            
            mock_app._render_theorem_selection_interface()
            
            # Verify navigation buttons were rendered (should be called multiple times)
            assert mock_button.called
    
    @pytest.mark.skip("Performance testing requires actual data - skip for unit tests")
    def test_performance_requirements(self, mock_app, mock_theorem):
        """Test that performance requirements are met."""
        import time
        
        with patch.object(mock_app, '_get_selected_theorem_for_proof', return_value=mock_theorem), \
             patch('streamlit.session_state', MockSessionState()):
            
            # Mock proof visualizer with timing
            mock_app.proof_visualizer.render_proof_visualization = Mock()
            
            start_time = time.time()
            mock_app.render_proof_viewer_page()
            end_time = time.time()
            
            # Verify rendering completes quickly (should be much less than 3 seconds for mocked components)
            assert (end_time - start_time) < 1.0
    
    def test_error_recovery_and_logging(self, mock_app, mock_theorem):
        """Test error recovery and logging functionality."""
        with patch.object(mock_app, '_get_selected_theorem_for_proof', return_value=mock_theorem), \
             patch('streamlit.session_state', MockSessionState()), \
             patch('streamlit.error'), \
             patch('streamlit.info'):
            
            # Mock logger
            mock_app.logger = Mock()
            
            # Simulate error in proof visualization
            mock_app.proof_visualizer.render_proof_visualization = Mock(side_effect=Exception("Test error"))
            
            mock_app.render_proof_viewer_page()
            
            # Verify error was logged
            mock_app.logger.error.assert_called()
    
    def test_memory_management(self, mock_app, mock_theorem):
        """Test that memory management works correctly for proof sessions."""
        with patch('streamlit.session_state', MockSessionState()) as mock_session_state, \
             patch.object(mock_app, '_get_selected_theorem_for_proof', return_value=mock_theorem):
            
            mock_app.proof_visualizer.render_proof_visualization = Mock()
            
            # Render multiple times to test session reuse
            mock_app.render_proof_viewer_page()
            mock_app.render_proof_viewer_page()
            
            # Verify only one session was created
            session_keys = [key for key in mock_session_state.keys() if key.startswith('proof_session_')]
            assert len(session_keys) == 1 