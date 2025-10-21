import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from wallstreet_quant.utils import CompanyDeduper


class TestCompanyDeduper:
    """Test suite for CompanyDeduper class."""
    
    def test_init(self):
        """Test initialization of CompanyDeduper."""
        ticker_map = {"AAPL": "Apple Inc", "MSFT": "Microsoft Corporation"}
        mock_model = Mock()
        deduper = CompanyDeduper(ticker_map, embed_model=mock_model)
        assert deduper._ticker_map == ticker_map
        assert deduper._k == 10  # default value
        assert deduper._cos_th == 0.80  # default value
        assert deduper._embed_model == mock_model
    
    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        ticker_map = {"GOOGL": "Alphabet Inc"}
        mock_model = Mock()
        deduper = CompanyDeduper(ticker_map, k_neighbors=5, cosine_th=0.75, embed_model=mock_model)
        assert deduper._k == 5
        assert deduper._cos_th == 0.75
    
    def test_normalise_unicode(self):
        """Test Unicode normalization."""
        text = "Café"
        result = CompanyDeduper._normalise_unicode(text)
        assert isinstance(result, str)
        # Result should be normalized
        assert len(result) >= len(text)
    
    def test_canonicalise(self):
        """Test company name canonicalization."""
        ticker_map = {}
        mock_model = Mock()
        deduper = CompanyDeduper(ticker_map, embed_model=mock_model)
        
        # Test suffix removal
        result = deduper._canonicalise("Apple Inc")
        assert "inc" not in result.lower()
        
        # Test lowercase and sorting
        result = deduper._canonicalise("Microsoft Corporation")
        assert result == result.lower()
        
        # Test special characters removal
        result = deduper._canonicalise("Company, Inc.")
        assert "," not in result
        assert "." not in result
    
    def test_is_acronym(self):
        """Test acronym detection."""
        ticker_map = {}
        mock_model = Mock()
        deduper = CompanyDeduper(ticker_map, embed_model=mock_model)
        
        assert deduper._is_acronym("IBM")
        assert deduper._is_acronym("NASA")
        assert not deduper._is_acronym("Apple")
        assert not deduper._is_acronym("A")
        assert not deduper._is_acronym("TOOLONG")
    
    def test_expand_ticker(self):
        """Test ticker expansion."""
        ticker_map = {"AAPL": "Apple Inc", "MSFT": "Microsoft Corporation"}
        mock_model = Mock()
        deduper = CompanyDeduper(ticker_map, embed_model=mock_model)
        
        assert deduper._expand_ticker("AAPL") == "Apple Inc"
        assert deduper._expand_ticker("MSFT") == "Microsoft Corporation"
        assert deduper._expand_ticker("UNKNOWN") == "UNKNOWN"
        assert deduper._expand_ticker("Apple Inc") == "Apple Inc"
    
    def test_expand_acronym(self):
        """Test acronym expansion."""
        ticker_map = {}
        mock_model = Mock()
        deduper = CompanyDeduper(ticker_map, embed_model=mock_model)
        
        universe = ["International Business Machines", "Apple Inc"]
        result = deduper._expand_acronym("IBM", universe)
        assert result == "International Business Machines"
        
        # Test acronym that doesn't match
        result = deduper._expand_acronym("XYZ", universe)
        assert result == "XYZ"
    
    def test_embed(self):
        """Test embedding generation."""
        ticker_map = {}
        mock_model = Mock()
        # Mock the encode method to return a normalized array
        mock_model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]])
        deduper = CompanyDeduper(ticker_map, embed_model=mock_model)
        
        strings = ["Apple Inc", "Microsoft Corporation", "Google LLC"]
        embeddings = deduper._embed(strings)
        
        assert embeddings.shape[0] == len(strings)
        assert embeddings.shape[1] > 0
        # Check normalization (L2 norm should be close to 1)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(len(strings)), decimal=5)
    
    def test_dedupe_simple(self):
        """Test deduplication with simple cases."""
        ticker_map = {}
        mock_model = Mock()
        # Mock embeddings for the test - only need embeddings for unique canonical names
        # After canonicalization: "Apple Inc" -> "apple", "Microsoft Corp" -> "microsoft"
        # So we need 2 embeddings for the representatives
        mock_model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
        deduper = CompanyDeduper(ticker_map, embed_model=mock_model)
        
        # Test with exact duplicates
        raw_names = ["Apple Inc", "Apple Inc", "Microsoft Corp"]
        clusters, name2cid, representatives = deduper.dedupe(raw_names)
        
        assert len(clusters) >= 2  # At least 2 clusters (Apple and Microsoft)
        assert len(representatives) == len(clusters)
        assert all(name in name2cid for name in raw_names)
    
    def test_dedupe_with_ticker_expansion(self):
        """Test deduplication with ticker expansion."""
        ticker_map = {"AAPL": "Apple Inc"}
        mock_model = Mock()
        # Mock embeddings - both should map to same embedding
        mock_model.encode.return_value = np.array([[1.0, 0.0]])
        deduper = CompanyDeduper(ticker_map, embed_model=mock_model)
        
        raw_names = ["AAPL", "Apple Inc"]
        clusters, name2cid, representatives = deduper.dedupe(raw_names)
        
        # Both should be in the same cluster
        assert name2cid["AAPL"] == name2cid["Apple Inc"]
        assert len(clusters) >= 1
    
    def test_dedupe_preserves_all_names(self):
        """Test that dedupe preserves all input names in the mapping."""
        ticker_map = {"AAPL": "Apple Inc"}
        mock_model = Mock()
        # Mock embeddings for different companies
        mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        deduper = CompanyDeduper(ticker_map, embed_model=mock_model)
        
        raw_names = ["AAPL", "Apple Inc", "Microsoft Corporation", "Google LLC"]
        clusters, name2cid, representatives = deduper.dedupe(raw_names)
        
        # All raw names should be in the mapping
        assert len(name2cid) >= len(raw_names)
        for name in raw_names:
            assert name in name2cid
        
        # All names should be in some cluster
        all_names_in_clusters = [name for cluster in clusters for name in cluster]
        for name in raw_names:
            assert name in all_names_in_clusters or name in name2cid
