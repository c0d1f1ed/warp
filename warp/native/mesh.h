/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include "builtin.h"
#include "bvh.h"
#include "intersect.h"
#include "array.h"

#define BVH_DEBUG 0

namespace wp
{

struct Mesh
{
    array_t<vec3> points;
	array_t<vec3> velocities;

    array_t<int> indices;

	bounds3* bounds;

    int num_points;
    int num_tris;

    BVH bvh;

	void* context;

    inline CUDA_CALLABLE Mesh(int id = 0) {
		// for backward a = 0 initialization syntax
		bounds = nullptr;
		num_points = 0;
		num_tris = 0;
		context = nullptr;
	}    
};

CUDA_CALLABLE inline Mesh mesh_get(uint64_t id)
{
    return *(Mesh*)(id);
}

CUDA_CALLABLE inline void adj_mesh_get(uint64_t id, uint64_t& adj_id, const Mesh& adj_ret)
{
	// no-op
}


CUDA_CALLABLE inline Mesh& operator += (Mesh& a, const Mesh& b) {
	// dummy operator needed for adj_select involving meshes
	return a;
}

CUDA_CALLABLE inline float distance_to_aabb_sq(const vec3& p, const vec3& lower, const vec3& upper)
{
	vec3 cp = closest_point_to_aabb(p, lower, upper);

	return length_sq(p-cp);
}


// fwd
CUDA_CALLABLE inline float mesh_query_inside(uint64_t id, const vec3& p);


// returns true if there is a point (strictly) < distance max_dist
CUDA_CALLABLE inline bool mesh_query_point(uint64_t id, const vec3& point, float max_dist, float& inside, int& face, float& u, float& v)
{
    Mesh mesh = mesh_get(id);

	if (mesh.bvh.num_nodes == 0)
		return false;

	int stack[32];
    stack[0] = mesh.bvh.root;

	int count = 1;

	float min_dist_sq = max_dist*max_dist;
	int min_face;
	float min_v;
	float min_w;

#if BVH_DEBUG
	int tests = 0;
	int secondary_culls = 0;

	std::vector<int> test_history;
	std::vector<vec3> test_centers;
	std::vector<vec3> test_extents;
#endif

	while (count)
	{
		const int nodeIndex = stack[--count];

		BVHPackedNodeHalf lower = mesh.bvh.node_lowers[nodeIndex];
		BVHPackedNodeHalf upper = mesh.bvh.node_uppers[nodeIndex];
	
		// re-test distance
		float node_dist_sq = distance_to_aabb_sq(point, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z));
		if (node_dist_sq > min_dist_sq)
		{
#if BVH_DEBUG			
			secondary_culls++;
#endif			
			continue;
		}

		const int left_index = lower.i;
		const int right_index = upper.i;

		if (lower.b)
		{	
			// compute closest point on tri
			int i = mesh.indices[left_index*3+0];
			int j = mesh.indices[left_index*3+1];
			int k = mesh.indices[left_index*3+2];

			vec3 p = mesh.points[i];
			vec3 q = mesh.points[j];
			vec3 r = mesh.points[k];
			
			vec3 e0 = q-p;
			vec3 e1 = r-p;
			vec3 e2 = r-q;
			vec3 normal = cross(e0, e1);
			
			// sliver detection
			if (length(normal)/(dot(e0,e0) + dot(e1,e1) + dot(e2,e2)) < 1.e-6f)
				continue;

			vec2 barycentric = closest_point_to_triangle(p, q, r, point);
			float u = barycentric[0];
			float v = barycentric[1];
			float w = 1.f - u - v;
			vec3 c = u*p + v*q + w*r;

			float dist_sq = length_sq(c-point);

			if (dist_sq < min_dist_sq)
			{
				min_dist_sq = dist_sq;
				min_v = v;
				min_w = w;
				min_face = left_index;
			}

#if BVH_DEBUG

			tests++;

			bounds3 b;
			b = bounds_union(b, p);
			b = bounds_union(b, q);
			b = bounds_union(b, r);

			if (distance_to_aabb_sq(point, b.lower, b.upper) < max_dist*max_dist)
			{
				//if (dist_sq < max_dist*max_dist)
				test_history.push_back(left_index);
				test_centers.push_back(b.center());
				test_extents.push_back(b.edges());
			}
#endif

		}
		else
		{
			BVHPackedNodeHalf left_lower = mesh.bvh.node_lowers[left_index];
			BVHPackedNodeHalf left_upper = mesh.bvh.node_uppers[left_index];

			BVHPackedNodeHalf right_lower = mesh.bvh.node_lowers[right_index];
			BVHPackedNodeHalf right_upper = mesh.bvh.node_uppers[right_index];

			float left_dist_sq = distance_to_aabb_sq(point, vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z));
			float right_dist_sq = distance_to_aabb_sq(point, vec3(right_lower.x, right_lower.y, right_lower.z), vec3(right_upper.x, right_upper.y, right_upper.z));


			// traverse smaller area first
			//float left_score = bounds3(vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z)).area();
			//float right_score = bounds3(vec3(right_lower.x, right_lower.y, right_lower.z), vec3(right_upper.x, right_upper.y, right_upper.z)).area();

			float left_score = left_dist_sq;
			float right_score = right_dist_sq;

			// tie break based on distance to centroid if point inside both bounds
			// if (left_score == right_score)
			// {
			// 	// use distance from box centroid to order children
			// 	//left_score = -length_sq(point-bounds3(vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z)).center());
			// 	//right_score = -length_sq(point-bounds3(vec3(right_lower.x, right_lower.y, right_lower.z), vec3(right_upper.x, right_upper.y, right_upper.z)).center());
			// 	left_score = bounds3(vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z)).area();
			// 	right_score = bounds3(vec3(right_lower.x, right_lower.y, right_lower.z), vec3(right_upper.x, right_upper.y, right_upper.z)).area();

			// }

			//if (left_dist_sq < right_dist_sq)
			if (left_score < right_score)
			{
				// put left on top of the stack
				if (right_dist_sq < min_dist_sq)
					stack[count++] = right_index;

				if (left_dist_sq < min_dist_sq)
					stack[count++] = left_index;
			}
			else
			{
				// put right on top of the stack
				if (left_dist_sq < min_dist_sq)
					stack[count++] = left_index;

				if (right_dist_sq < min_dist_sq)
					stack[count++] = right_index;
			}
		}
	}


#if BVH_DEBUG
	printf("%d\n", tests);

    static int max_tests = 0;
	static vec3 max_point;
	static float max_point_dist = 0.0f;
	static int max_secondary_culls = 0;

	if (secondary_culls > max_secondary_culls)
		max_secondary_culls = secondary_culls;

    if (tests > max_tests)
	{
        max_tests = tests;
		max_point = point;
		max_point_dist = sqrtf(min_dist_sq);

		printf("max_tests: %d max_point: %f %f %f max_point_dist: %f max_second_culls: %d\n", max_tests, max_point[0], max_point[1], max_point[2], max_point_dist, max_secondary_culls);

		FILE* f = fopen("test_history.txt", "w");
		for (int i=0; i < test_history.size(); ++i)
		{
			fprintf(f, "%d, %f, %f, %f, %f, %f, %f\n", 
			test_history[i], 
			test_centers[i][0], test_centers[i][1], test_centers[i][2],
			test_extents[i][0], test_extents[i][1], test_extents[i][2]);
		}

		fclose(f);
	}
#endif

	// check if we found a point, and write outputs
	if (min_dist_sq < max_dist*max_dist)
	{
		u = 1.0f - min_v - min_w;
		v = min_v;
		face = min_face;
		
		// determine inside outside using ray-cast parity check
		inside = mesh_query_inside(id, point);
		
		return true;
	}
	else
	{
		return false;
	}
}

CUDA_CALLABLE inline void adj_mesh_query_point(uint64_t id, const vec3& point, float max_dist, float& inside, int& face, float& u, float& v,
											   uint64_t adj_id, vec3& adj_point, float& adj_max_dist, float& adj_inside, int& adj_face, float& adj_u, float& adj_v, bool& adj_ret)
{
	Mesh mesh = mesh_get(id);
	
	// face is determined by BVH in forward pass
	int i = mesh.indices[face*3+0];
	int j = mesh.indices[face*3+1];
	int k = mesh.indices[face*3+2];

	vec3 p = mesh.points[i];
	vec3 q = mesh.points[j];
	vec3 r = mesh.points[k];

	vec3 adj_p, adj_q, adj_r;

	vec2 adj_uv(adj_u, adj_v);

	adj_closest_point_to_triangle(p, q, r, point, adj_p, adj_q, adj_r, adj_point, adj_uv);
}


CUDA_CALLABLE inline bool mesh_query_ray(uint64_t id, const vec3& start, const vec3& dir, float max_t, float& t, float& u, float& v, float& sign, vec3& normal, int& face)
{
    Mesh mesh = mesh_get(id);

	if (mesh.bvh.num_nodes == 0)
		return false;

    int stack[32];
	stack[0] = mesh.bvh.root;
	int count = 1;

	vec3 rcp_dir = vec3(1.0f/dir[0], 1.0f/dir[1], 1.0f/dir[2]);

	float min_t = max_t;
	int min_face;
	float min_u;
	float min_v;
	float min_sign = 1.0f;
	vec3 min_normal;

	while (count)
	{
		const int nodeIndex = stack[--count];

		BVHPackedNodeHalf lower = mesh.bvh.node_lowers[nodeIndex];
		BVHPackedNodeHalf upper = mesh.bvh.node_uppers[nodeIndex];

		// todo: switch to robust ray-aabb, or expand bounds in build stage
		float eps = 1.e-3f;
		float t = 0.0f;
		bool hit = intersect_ray_aabb(start, rcp_dir, vec3(lower.x-eps, lower.y-eps, lower.z-eps), vec3(upper.x+eps, upper.y+eps, upper.z+eps), t);

		if (hit && t < min_t)
		{
			const int left_index = lower.i;
			const int right_index = upper.i;

			if (lower.b)
			{	
				// compute closest point on tri
				int i = mesh.indices[left_index*3+0];
				int j = mesh.indices[left_index*3+1];
				int k = mesh.indices[left_index*3+2];

				vec3 p = mesh.points[i];
				vec3 q = mesh.points[j];
				vec3 r = mesh.points[k];

				float t, u, v, sign;
				vec3 n;
				
				if (intersect_ray_tri_woop(start, dir, p, q, r, t, u, v, sign, &n))
				{
					if (t < min_t && t >= 0.0f)
					{
						min_t = t;
						min_face = left_index;
						min_u = u;
						min_v = v;
						min_sign = sign;
						min_normal = n;
					}
				}
			}
			else
			{
				stack[count++] = left_index;
				stack[count++] = right_index;
			}
		}
	}

	if (min_t < max_t)
	{
		// write outputs
		u = min_u;
		v = min_v;
		sign = min_sign;
		t = min_t;
		normal = normalize(min_normal);
		face = min_face;

		return true;
	}
	else
	{
		return false;
	}
	
}


CUDA_CALLABLE inline void adj_mesh_query_ray(
	uint64_t id, const vec3& start, const vec3& dir, float max_t, float& t, float& u, float& v, float& sign, vec3& n, int& face,
	uint64_t adj_id, vec3& adj_start, vec3& adj_dir, float& adj_max_t, float& adj_t, float& adj_u, float& adj_v, float& adj_sign, vec3& adj_n, int& adj_face, bool adj_ret)
{

	Mesh mesh = mesh_get(id);
	
	// face is determined by BVH in forward pass
	int i = mesh.indices[face*3+0];
	int j = mesh.indices[face*3+1];
	int k = mesh.indices[face*3+2];

	vec3 a = mesh.points[i];
	vec3 b = mesh.points[j];
	vec3 c = mesh.points[k];

	vec3 adj_a, adj_b, adj_c;

	adj_intersect_ray_tri_woop(start, dir, a, b, c, t, u, v, sign, &n, adj_start, adj_dir, adj_a, adj_b, adj_c, adj_t, adj_u, adj_v, adj_sign, &adj_n, adj_ret);

}


// determine if a point is inside (ret < 0 ) or outside the mesh (ret > 0)
CUDA_CALLABLE inline float mesh_query_inside(uint64_t id, const vec3& p)
{
    float t, u, v, sign;
	vec3 n;
	int face;

	int vote = 0;

	for(int i = 0; i <3; ++i){
		if (mesh_query_ray(id, p, vec3(float(i==0), float(i==1), float(i==2)), FLT_MAX, t, u, v, sign, n, face) && sign < 0) {
			vote++;
		}
	}

	if (vote == 3)
		return -1.0f;
	else
		return 1.0f;
}


// stores state required to traverse the BVH nodes that 
// overlap with a query AABB.
struct mesh_query_aabb_t
{
    CUDA_CALLABLE mesh_query_aabb_t()
    {
    }
    CUDA_CALLABLE mesh_query_aabb_t(int)
    {
    } // for backward pass

    // Mesh Id
    Mesh mesh;
	// BVH traversal stack:
	int stack[32];
	int count;

    // inputs
    wp::vec3 input_lower;
    wp::vec3 input_upper;

	// Face
	int face;
};



CUDA_CALLABLE inline mesh_query_aabb_t mesh_query_aabb(
    uint64_t id, const vec3& lower, const vec3& upper)
{
    // This routine traverses the BVH tree until it finds
	// the first triangle with an overlapping bvh. 

    // initialize empty
	mesh_query_aabb_t query;
	query.face = -1;

	Mesh mesh = mesh_get(id);

	query.mesh = mesh;
	
    // if no bvh nodes, return empty query.
    if (mesh.bvh.num_nodes == 0)
    {
		query.count = 0;
		return query;
	}

    // optimization: make the latest
	
	query.stack[0] = mesh.bvh.root;
	query.count = 1;
    query.input_lower = lower;
    query.input_upper = upper;

    wp::bounds3 input_bounds(query.input_lower, query.input_upper);
	
    // Navigate through the bvh, find the first overlapping leaf node.
    while (query.count)
    {
		const int nodeIndex = query.stack[--query.count];
		BVHPackedNodeHalf node_lower = mesh.bvh.node_lowers[nodeIndex];
		BVHPackedNodeHalf node_upper = mesh.bvh.node_uppers[nodeIndex];

		wp::vec3 lower_pos(node_lower.x, node_lower.y, node_lower.z);
		wp::vec3 upper_pos(node_upper.x, node_upper.y, node_upper.z);
        wp::bounds3 current_bounds(lower_pos, upper_pos);
        if (!input_bounds.overlaps(current_bounds))
        {
            // Skip this box, it doesn't overlap with our target box.
			continue;
		}

		const int left_index = node_lower.i;
		const int right_index = node_upper.i;

        // Make bounds from this AABB
        if (node_lower.b)
        {
			// found very first triangle index.
			// Back up one level and return 
			query.stack[query.count++] = nodeIndex;
			return query;
        }
        else
        {	
		  query.stack[query.count++] = left_index;
		  query.stack[query.count++] = right_index;
		}
	}	

	return query;
}

//Stub
CUDA_CALLABLE inline void adj_mesh_query_aabb(uint64_t id, const vec3& lower, const vec3& upper,
											   uint64_t, vec3&, vec3&, mesh_query_aabb_t&)
{

}

CUDA_CALLABLE inline bool mesh_query_aabb_next(mesh_query_aabb_t& query, int& index)
{
    Mesh mesh = query.mesh;
	
	wp::bounds3 input_bounds(query.input_lower, query.input_upper);
    // Navigate through the bvh, find the first overlapping leaf node.
    while (query.count)
    {
        const int nodeIndex = query.stack[--query.count];
        BVHPackedNodeHalf node_lower = mesh.bvh.node_lowers[nodeIndex];
        BVHPackedNodeHalf node_upper = mesh.bvh.node_uppers[nodeIndex];

        wp::vec3 lower_pos(node_lower.x, node_lower.y, node_lower.z);
        wp::vec3 upper_pos(node_upper.x, node_upper.y, node_upper.z);
        wp::bounds3 current_bounds(lower_pos, upper_pos);
        if (!input_bounds.overlaps(current_bounds))
        {
            // Skip this box, it doesn't overlap with our target box.
            continue;
        }

        const int left_index = node_lower.i;
        const int right_index = node_upper.i;

        // Make bounds from this AABB
        if (node_lower.b)
        {
            // found very first triangle index
            query.face = left_index;
			index = left_index;
            return true;
        }
        else
        {

            query.stack[query.count++] = left_index;
            query.stack[query.count++] = right_index;
        }
    }
    return false;
}


CUDA_CALLABLE inline int iter_next(mesh_query_aabb_t& query)
{
    return query.face;
}

CUDA_CALLABLE inline bool iter_cmp(mesh_query_aabb_t& query)
{
    bool finished = mesh_query_aabb_next(query, query.face);
    return finished;
}

CUDA_CALLABLE inline mesh_query_aabb_t iter_reverse(const mesh_query_aabb_t& query)
{
    // can't reverse BVH queries, users should not rely on neighbor ordering
    return query;
}


// stub
CUDA_CALLABLE inline void adj_mesh_query_aabb_next(mesh_query_aabb_t& query, int& index, mesh_query_aabb_t&, int&, bool&) 
{

}


CUDA_CALLABLE inline vec3 mesh_eval_position(uint64_t id, int tri, float u, float v)
{
	Mesh mesh = mesh_get(id);

	if (!mesh.points)
		return vec3();

	assert(tri < mesh.num_tris);

	int i = mesh.indices[tri*3+0];
	int j = mesh.indices[tri*3+1];
	int k = mesh.indices[tri*3+2];

	vec3 p = mesh.points[i];
	vec3 q = mesh.points[j];
	vec3 r = mesh.points[k];

	return p*u + q*v + r*(1.0f-u-v);
}

CUDA_CALLABLE inline vec3 mesh_eval_velocity(uint64_t id, int tri, float u, float v)
{
	Mesh mesh = mesh_get(id);

	if (!mesh.velocities)
		return vec3();

	assert(tri < mesh.num_tris);

	int i = mesh.indices[tri*3+0];
	int j = mesh.indices[tri*3+1];
	int k = mesh.indices[tri*3+2];

	vec3 vp = mesh.velocities[i];
	vec3 vq = mesh.velocities[j];
	vec3 vr = mesh.velocities[k];

	return vp*u + vq*v + vr*(1.0f-u-v);
}


CUDA_CALLABLE inline void adj_mesh_eval_position(uint64_t id, int tri, float u, float v,
												 uint64_t& adj_id, int& adj_tri, float& adj_u, float& adj_v, const vec3& adj_ret)
{
	Mesh mesh = mesh_get(id);

	if (!mesh.points)
		return;

	assert(tri < mesh.num_tris);

	int i = mesh.indices[tri*3+0];
	int j = mesh.indices[tri*3+1];
	int k = mesh.indices[tri*3+2];

	vec3 p = mesh.points[i];
	vec3 q = mesh.points[j];
	vec3 r = mesh.points[k];

	adj_u += (p[0] - r[0]) * adj_ret[0] + (p[1] - r[1]) * adj_ret[1] + (p[2] - r[2]) * adj_ret[2];
	adj_v += (q[0] - r[0]) * adj_ret[0] + (q[1] - r[1]) * adj_ret[1] + (q[2] - r[2]) * adj_ret[2];
}

CUDA_CALLABLE inline void adj_mesh_eval_velocity(uint64_t id, int tri, float u, float v,
												 uint64_t& adj_id, int& adj_tri, float& adj_u, float& adj_v, const vec3& adj_ret)
{
	Mesh mesh = mesh_get(id);

	if (!mesh.velocities)
		return;

	assert(tri < mesh.num_tris);

	int i = mesh.indices[tri*3+0];
	int j = mesh.indices[tri*3+1];
	int k = mesh.indices[tri*3+2];

	vec3 vp = mesh.velocities[i];
	vec3 vq = mesh.velocities[j];
	vec3 vr = mesh.velocities[k];

	adj_u += (vp[0] - vr[0]) * adj_ret[0] + (vp[1] - vr[1]) * adj_ret[1] + (vp[2] - vr[2]) * adj_ret[2];
	adj_v += (vq[0] - vr[0]) * adj_ret[0] + (vq[1] - vr[1]) * adj_ret[1] + (vq[2] - vr[2]) * adj_ret[2];
}

CUDA_CALLABLE inline vec3 mesh_eval_face_normal(uint64_t id, int tri)
{
	Mesh mesh = mesh_get(id);

	if (!mesh.points)
		return vec3();

	assert(tri < mesh.num_tris);

	int i = mesh.indices[tri*3+0];
	int j = mesh.indices[tri*3+1];
	int k = mesh.indices[tri*3+2];

	vec3 p = mesh.points[i];
	vec3 q = mesh.points[j];
	vec3 r = mesh.points[k];

	return normalize(cross(q - p, r - p));
}

CUDA_CALLABLE inline void adj_mesh_eval_face_normal(uint64_t id, int tri,
												    uint64_t& adj_id, int& adj_tri, const vec3& adj_ret)
{
	// no-op
}

CUDA_CALLABLE inline vec3 mesh_get_point(uint64_t id, int index)
{
	Mesh mesh = mesh_get(id);

	if (!mesh.points)
		return vec3();

#if FP_CHECK
	if (index >= mesh.num_tris * 3)
	{
		printf("mesh_get_point (%llu, %d) out of bounds at %s:%d\n", id, index, __FILE__, __LINE__);
		assert(0);
	}
#endif

	int i = mesh.indices[index];
	return mesh.points[i];
}

CUDA_CALLABLE inline void adj_mesh_get_point(uint64_t id, int index,
										     uint64_t& adj_id, int& adj_index, const vec3& adj_ret)
{
	// no-op
}

CUDA_CALLABLE inline vec3 mesh_get_velocity(uint64_t id, int index)
{
	Mesh mesh = mesh_get(id);

	if (!mesh.velocities)
		return vec3();

#if FP_CHECK
	if (index >= mesh.num_tris * 3)
	{
		printf("mesh_get_velocity (%llu, %d) out of bounds at %s:%d\n", id, index, __FILE__, __LINE__);
		assert(0);
	}
#endif

	int i = mesh.indices[index];
	return mesh.velocities[i];
}

CUDA_CALLABLE inline void adj_mesh_get_velocity(uint64_t id, int index,
										     uint64_t& adj_id, int& adj_index, const vec3& adj_ret)
{
	// no-op
}

CUDA_CALLABLE inline int mesh_get_index(uint64_t id, int face_vertex_index)
{
	Mesh mesh = mesh_get(id);

	if (!mesh.indices)
		return -1;

	assert(face_vertex_index < mesh.num_tris * 3);

	return mesh.indices[face_vertex_index];
}

CUDA_CALLABLE inline void adj_mesh_get_index(uint64_t id, int index,
										     uint64_t& adj_id, int& adj_index, const vec3& adj_ret)
{
	// no-op
}

bool mesh_get_descriptor(uint64_t id, Mesh& mesh);
void mesh_add_descriptor(uint64_t id, const Mesh& mesh);
void mesh_rem_descriptor(uint64_t id);


} // namespace wp
