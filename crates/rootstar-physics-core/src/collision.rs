//! Collision detection primitives for VR neural mapping.
//!
//! Provides basic collision detection between environmental effects and body mesh.

use serde::{Deserialize, Serialize};

use crate::types::Velocity;

/// Axis-aligned bounding box for broad-phase collision.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    /// Minimum corner.
    pub min: Velocity,
    /// Maximum corner.
    pub max: Velocity,
}

impl BoundingBox {
    /// Create a new bounding box.
    #[must_use]
    pub fn new(min: Velocity, max: Velocity) -> Self {
        Self {
            min: Velocity::new(min.x.min(max.x), min.y.min(max.y), min.z.min(max.z)),
            max: Velocity::new(min.x.max(max.x), min.y.max(max.y), min.z.max(max.z)),
        }
    }

    /// Create a bounding box centered at origin with given half-extents.
    #[must_use]
    pub fn from_center_extents(center: Velocity, half_extents: Velocity) -> Self {
        Self {
            min: Velocity::new(
                center.x - half_extents.x,
                center.y - half_extents.y,
                center.z - half_extents.z,
            ),
            max: Velocity::new(
                center.x + half_extents.x,
                center.y + half_extents.y,
                center.z + half_extents.z,
            ),
        }
    }

    /// Get the center of the bounding box.
    #[must_use]
    pub fn center(&self) -> Velocity {
        Velocity::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    /// Get the half-extents (size from center to edge).
    #[must_use]
    pub fn half_extents(&self) -> Velocity {
        Velocity::new(
            (self.max.x - self.min.x) * 0.5,
            (self.max.y - self.min.y) * 0.5,
            (self.max.z - self.min.z) * 0.5,
        )
    }

    /// Check if a point is inside the bounding box.
    #[must_use]
    pub fn contains(&self, point: &Velocity) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Check if this bounding box intersects another.
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Expand the bounding box to include a point.
    #[must_use]
    pub fn expand(&self, point: &Velocity) -> Self {
        Self {
            min: Velocity::new(
                self.min.x.min(point.x),
                self.min.y.min(point.y),
                self.min.z.min(point.z),
            ),
            max: Velocity::new(
                self.max.x.max(point.x),
                self.max.y.max(point.y),
                self.max.z.max(point.z),
            ),
        }
    }

    /// Merge with another bounding box.
    #[must_use]
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            min: Velocity::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Velocity::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }
}

impl Default for BoundingBox {
    fn default() -> Self {
        Self {
            min: Velocity::zero(),
            max: Velocity::zero(),
        }
    }
}

/// A ray for ray-casting against mesh.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Ray {
    /// Origin point.
    pub origin: Velocity,
    /// Direction (should be normalized).
    pub direction: Velocity,
}

impl Ray {
    /// Create a new ray.
    #[must_use]
    pub fn new(origin: Velocity, direction: Velocity) -> Self {
        Self {
            origin,
            direction: direction.normalized(),
        }
    }

    /// Get a point along the ray at distance t.
    #[must_use]
    pub fn point_at(&self, t: f32) -> Velocity {
        Velocity::new(
            self.origin.x + self.direction.x * t,
            self.origin.y + self.direction.y * t,
            self.origin.z + self.direction.z * t,
        )
    }

    /// Check intersection with a bounding box.
    ///
    /// Returns the entry and exit distances, or None if no intersection.
    #[must_use]
    pub fn intersect_aabb(&self, aabb: &BoundingBox) -> Option<(f32, f32)> {
        let inv_dir = Velocity::new(
            if self.direction.x.abs() > 1e-8 {
                1.0 / self.direction.x
            } else {
                f32::MAX
            },
            if self.direction.y.abs() > 1e-8 {
                1.0 / self.direction.y
            } else {
                f32::MAX
            },
            if self.direction.z.abs() > 1e-8 {
                1.0 / self.direction.z
            } else {
                f32::MAX
            },
        );

        let t1 = (aabb.min.x - self.origin.x) * inv_dir.x;
        let t2 = (aabb.max.x - self.origin.x) * inv_dir.x;
        let t3 = (aabb.min.y - self.origin.y) * inv_dir.y;
        let t4 = (aabb.max.y - self.origin.y) * inv_dir.y;
        let t5 = (aabb.min.z - self.origin.z) * inv_dir.z;
        let t6 = (aabb.max.z - self.origin.z) * inv_dir.z;

        let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        if tmax >= tmin && tmax >= 0.0 {
            Some((tmin.max(0.0), tmax))
        } else {
            None
        }
    }
}

/// Result of a collision test.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CollisionResult {
    /// Whether a collision occurred.
    pub hit: bool,
    /// Distance to the collision point (if hit).
    pub distance: f32,
    /// Position of the collision point.
    pub point: Velocity,
    /// Normal at the collision point.
    pub normal: Velocity,
    /// UV coordinates at the collision point (for body region mapping).
    pub uv: (f32, f32),
}

impl CollisionResult {
    /// Create a miss result.
    #[must_use]
    pub fn miss() -> Self {
        Self {
            hit: false,
            distance: f32::MAX,
            point: Velocity::zero(),
            normal: Velocity::zero(),
            uv: (0.0, 0.0),
        }
    }

    /// Create a hit result.
    #[must_use]
    pub fn hit(distance: f32, point: Velocity, normal: Velocity, uv: (f32, f32)) -> Self {
        Self {
            hit: true,
            distance,
            point,
            normal,
            uv,
        }
    }
}

impl Default for CollisionResult {
    fn default() -> Self {
        Self::miss()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_contains() {
        let bb = BoundingBox::new(Velocity::new(-1.0, -1.0, -1.0), Velocity::new(1.0, 1.0, 1.0));

        assert!(bb.contains(&Velocity::zero()));
        assert!(!bb.contains(&Velocity::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn test_bounding_box_intersects() {
        let bb1 = BoundingBox::new(Velocity::new(0.0, 0.0, 0.0), Velocity::new(2.0, 2.0, 2.0));

        let bb2 = BoundingBox::new(Velocity::new(1.0, 1.0, 1.0), Velocity::new(3.0, 3.0, 3.0));

        let bb3 = BoundingBox::new(Velocity::new(5.0, 5.0, 5.0), Velocity::new(6.0, 6.0, 6.0));

        assert!(bb1.intersects(&bb2));
        assert!(!bb1.intersects(&bb3));
    }

    #[test]
    fn test_ray_aabb_intersection() {
        let aabb = BoundingBox::new(Velocity::new(-1.0, -1.0, -1.0), Velocity::new(1.0, 1.0, 1.0));

        // Ray from outside, pointing at center
        let ray = Ray::new(Velocity::new(-5.0, 0.0, 0.0), Velocity::new(1.0, 0.0, 0.0));

        let result = ray.intersect_aabb(&aabb);
        assert!(result.is_some());

        let (tmin, tmax) = result.unwrap();
        assert!((tmin - 4.0).abs() < 0.01);
        assert!((tmax - 6.0).abs() < 0.01);

        // Ray pointing away
        let ray_away = Ray::new(Velocity::new(-5.0, 0.0, 0.0), Velocity::new(-1.0, 0.0, 0.0));

        assert!(ray_away.intersect_aabb(&aabb).is_none());
    }

    #[test]
    fn test_ray_point_at() {
        let ray = Ray::new(Velocity::new(0.0, 0.0, 0.0), Velocity::new(1.0, 0.0, 0.0));

        let p = ray.point_at(5.0);
        assert!((p.x - 5.0).abs() < 0.01);
    }
}
