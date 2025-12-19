//! Subscript representation for einsum notation.

use alloc::vec::Vec;
use alloc::string::String;
use core::fmt;

/// A single index in an einsum subscript.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Index {
    /// A named index (a-z, A-Z).
    Named(char),
    /// Ellipsis representing zero or more batch dimensions.
    Ellipsis,
}

impl Index {
    /// Returns true if this is an ellipsis.
    #[inline]
    pub fn is_ellipsis(&self) -> bool {
        matches!(self, Index::Ellipsis)
    }

    /// Returns the character if this is a named index.
    #[inline]
    pub fn as_char(&self) -> Option<char> {
        match self {
            Index::Named(c) => Some(*c),
            Index::Ellipsis => None,
        }
    }
}

impl fmt::Display for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Index::Named(c) => write!(f, "{}", c),
            Index::Ellipsis => write!(f, "..."),
        }
    }
}

/// A subscript representing the indices of a single tensor.
///
/// For example, in `ij,jk->ik`, the subscripts are `ij`, `jk`, and `ik`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subscript {
    /// The indices in order.
    indices: Vec<Index>,
    /// Position of ellipsis (if any).
    ellipsis_pos: Option<usize>,
    /// Number of explicit (non-ellipsis) indices.
    explicit_count: usize,
}

impl Subscript {
    /// Creates an empty subscript.
    pub fn new() -> Self {
        Self {
            indices: Vec::new(),
            ellipsis_pos: None,
            explicit_count: 0,
        }
    }

    /// Creates a subscript from a list of indices.
    pub fn from_indices(indices: Vec<Index>) -> Self {
        let ellipsis_pos = indices.iter().position(|i| i.is_ellipsis());
        let explicit_count = indices.iter().filter(|i| !i.is_ellipsis()).count();
        Self {
            indices,
            ellipsis_pos,
            explicit_count,
        }
    }

    /// Creates a subscript from a string of characters.
    ///
    /// Each character becomes a named index. Use `...` for ellipsis.
    pub fn from_chars(chars: impl IntoIterator<Item = char>) -> Self {
        let indices: Vec<Index> = chars.into_iter().map(Index::Named).collect();
        Self::from_indices(indices)
    }

    /// Adds a named index.
    pub fn push_named(&mut self, c: char) {
        self.indices.push(Index::Named(c));
        self.explicit_count += 1;
    }

    /// Adds an ellipsis.
    pub fn push_ellipsis(&mut self) {
        if self.ellipsis_pos.is_none() {
            self.ellipsis_pos = Some(self.indices.len());
            self.indices.push(Index::Ellipsis);
        }
    }

    /// Returns true if this subscript has an ellipsis.
    #[inline]
    pub fn has_ellipsis(&self) -> bool {
        self.ellipsis_pos.is_some()
    }

    /// Returns the position of the ellipsis (if any).
    #[inline]
    pub fn ellipsis_position(&self) -> Option<usize> {
        self.ellipsis_pos
    }

    /// Returns the number of explicit (non-ellipsis) indices.
    #[inline]
    pub fn explicit_count(&self) -> usize {
        self.explicit_count
    }

    /// Returns the total number of index entries (including ellipsis as 1).
    #[inline]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns true if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Returns an iterator over the indices.
    pub fn iter(&self) -> impl Iterator<Item = &Index> {
        self.indices.iter()
    }

    /// Returns an iterator over named indices only (excluding ellipsis).
    pub fn named_indices(&self) -> impl Iterator<Item = char> + '_ {
        self.indices.iter().filter_map(|i| i.as_char())
    }

    /// Returns the indices as a slice.
    pub fn as_slice(&self) -> &[Index] {
        &self.indices
    }

    /// Checks if this subscript contains a specific named index.
    pub fn contains(&self, c: char) -> bool {
        self.indices.iter().any(|i| matches!(i, Index::Named(x) if *x == c))
    }

    /// Counts occurrences of a named index.
    pub fn count(&self, c: char) -> usize {
        self.indices.iter().filter(|i| matches!(i, Index::Named(x) if *x == c)).count()
    }

    /// Returns the position of a named index (first occurrence).
    pub fn position(&self, c: char) -> Option<usize> {
        self.indices.iter().position(|i| matches!(i, Index::Named(x) if *x == c))
    }

    /// Computes the actual number of dimensions given ellipsis dimension count.
    ///
    /// If there's no ellipsis, returns the explicit count.
    /// If there's an ellipsis, returns explicit_count + ellipsis_dims.
    pub fn ndims(&self, ellipsis_dims: usize) -> usize {
        if self.has_ellipsis() {
            self.explicit_count + ellipsis_dims
        } else {
            self.explicit_count
        }
    }

    /// Expands this subscript by replacing ellipsis with explicit batch indices.
    ///
    /// Returns a new subscript with no ellipsis.
    pub fn expand_ellipsis(&self, batch_indices: &[char]) -> Subscript {
        if !self.has_ellipsis() {
            return self.clone();
        }

        let mut expanded = Vec::with_capacity(self.explicit_count + batch_indices.len());
        for idx in &self.indices {
            match idx {
                Index::Ellipsis => {
                    for &c in batch_indices {
                        expanded.push(Index::Named(c));
                    }
                }
                other => expanded.push(*other),
            }
        }
        Subscript::from_indices(expanded)
    }

    /// Converts to a string representation.
    pub fn to_string(&self) -> String {
        let mut s = String::with_capacity(self.indices.len() * 2);
        for idx in &self.indices {
            match idx {
                Index::Named(c) => s.push(*c),
                Index::Ellipsis => s.push_str("..."),
            }
        }
        s
    }
}

impl Default for Subscript {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Subscript {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for idx in &self.indices {
            write!(f, "{}", idx)?;
        }
        Ok(())
    }
}

impl<'a> IntoIterator for &'a Subscript {
    type Item = &'a Index;
    type IntoIter = core::slice::Iter<'a, Index>;

    fn into_iter(self) -> Self::IntoIter {
        self.indices.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subscript_from_chars() {
        let sub = Subscript::from_chars(['i', 'j', 'k']);
        assert_eq!(sub.len(), 3);
        assert_eq!(sub.explicit_count(), 3);
        assert!(!sub.has_ellipsis());
        assert!(sub.contains('i'));
        assert!(sub.contains('j'));
        assert!(sub.contains('k'));
        assert!(!sub.contains('x'));
    }

    #[test]
    fn test_subscript_with_ellipsis() {
        let mut sub = Subscript::new();
        sub.push_ellipsis();
        sub.push_named('i');
        sub.push_named('j');

        assert_eq!(sub.len(), 3);
        assert_eq!(sub.explicit_count(), 2);
        assert!(sub.has_ellipsis());
        assert_eq!(sub.ellipsis_position(), Some(0));
    }

    #[test]
    fn test_expand_ellipsis() {
        let mut sub = Subscript::new();
        sub.push_ellipsis();
        sub.push_named('i');
        sub.push_named('j');

        let expanded = sub.expand_ellipsis(&['a', 'b']);
        assert_eq!(expanded.len(), 4);
        assert!(!expanded.has_ellipsis());
        assert!(expanded.contains('a'));
        assert!(expanded.contains('b'));
        assert!(expanded.contains('i'));
        assert!(expanded.contains('j'));
    }

    #[test]
    fn test_subscript_count() {
        let sub = Subscript::from_chars(['i', 'i', 'j']);
        assert_eq!(sub.count('i'), 2);
        assert_eq!(sub.count('j'), 1);
        assert_eq!(sub.count('k'), 0);
    }
}
