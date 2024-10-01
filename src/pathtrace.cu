#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define FIREST_BOUNCE 1
#define ERRORCHECK 1
#define STREAM_COMPACTION_INTERSECTION 1
#define SORT_BY_MATERIAL 1
#define STREAM_COMPACTION_PATH 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

// predicate that returns true when ray has no intersection
struct hasIntersection
{
	__host__ __device__
		bool operator()(const ShadeableIntersection& intersection) const
	{
        return (intersection.t >= 0.0f);
	}
};

// predicate that returns true when the path segment has no more bounces left
struct hasNoMoreBounces
{
    __host__ __device__
        bool operator()(const PathSegment& pathSegment) const
    {
        return (pathSegment.remainingBounces <= 0); 
    }
};

struct sortByMaterial
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const
    {
        return a.materialId < b.materialId; // Sort in ascending order by materialId
    }
};

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
//static Mesh* dev_meshes = NULL;
//static Triangle* dev_triangles = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

#if 1
size_t calculateMeshSize(const Mesh& mesh) {
    size_t size = sizeof(Mesh);
    /*size += mesh.vertices.size() * sizeof(glm::vec3);
    size += mesh.normals.size() * sizeof(glm::vec3); 
    size += mesh.uvs.size() * sizeof(glm::vec2);   
	size += mesh.indices.size() * sizeof(int);*/


    return size;
}
#endif

size_t calculateGeomSize(const Geom& geom) {
    size_t size = sizeof(Geom);

    // If the geom has a mesh, add the size of the mesh data
    if (geom.type == MESH && geom.mesh) {
        size += calculateMeshSize(*geom.mesh);
    }

    return size;
}

size_t calculateSceneSize(const std::vector<Geom>& geoms) {
    size_t totalSize = 0;

    // Add size of all Geom structures
    totalSize += geoms.size() * sizeof(Geom);

    // Add sizes for Mesh data where applicable
    for (const auto& geom : geoms) {
        if (geom.type == MESH && geom.mesh) {
            totalSize += calculateMeshSize(*geom.mesh);
        }
    }

    return totalSize;
}

void allocateMemForMesh(const Mesh& mesh, Mesh* dev_meshes) {
	// Allocate memory for vertices, normals, uvs, and triangles
    glm::vec3* dev_vertices = nullptr;
    glm::vec3* dev_normals = nullptr;
    glm::vec3* dev_uvs = nullptr;
    int* dev_indices = nullptr;
    Triangle* dev_triangles = nullptr;

    cudaMalloc(&dev_vertices, mesh.num_vertices * sizeof(glm::vec3));
    cudaMemcpy(dev_vertices, mesh.vertices, mesh.num_vertices * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_normals, mesh.num_normals * sizeof(glm::vec3));
	cudaMemcpy(dev_normals, mesh.normals, mesh.num_normals * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_uvs, mesh.num_uvs * sizeof(glm::vec3));
	cudaMemcpy(dev_uvs, mesh.uvs, mesh.num_uvs * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_indices, mesh.num_indices * sizeof(int));
	cudaMemcpy(dev_indices, mesh.indices, mesh.num_indices * sizeof(int), cudaMemcpyHostToDevice);

    if (mesh.num_triangles > 0) {
        // test mesh triangle data


        cudaMalloc(&dev_triangles, mesh.num_triangles * sizeof(Triangle));
        cudaMemcpy(dev_triangles, mesh.triangles, mesh.num_triangles * sizeof(Triangle), cudaMemcpyHostToDevice);

    }

	// Copy pointers to device memory
	cudaMemcpy(&(dev_meshes->vertices), &dev_vertices, sizeof(glm::vec3*), cudaMemcpyHostToDevice);

	cudaMemcpy(&(dev_meshes->normals), &dev_normals, sizeof(glm::vec3*), cudaMemcpyHostToDevice);

	cudaMemcpy(&(dev_meshes->uvs), &dev_uvs, sizeof(glm::vec3*), cudaMemcpyHostToDevice);

	cudaMemcpy(&(dev_meshes->indices), &dev_indices, sizeof(int*), cudaMemcpyHostToDevice);

	cudaMemcpy(&(dev_meshes->triangles), &dev_triangles, sizeof(Triangle*), cudaMemcpyHostToDevice);

	cudaMemcpy(&(dev_meshes->num_vertices), &mesh.num_vertices, sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(&(dev_meshes->num_normals), &mesh.num_normals, sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(&(dev_meshes->num_uvs), &mesh.num_uvs, sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(&(dev_meshes->num_indices), &mesh.num_indices, sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(&(dev_meshes->num_triangles), &mesh.num_triangles, sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("allocateMemForMesh: copy num_triangles to mesh device");
	
}

void allocateMemForTexture(const Texture& texture, Texture* dev_textures) {
    // Allocate memory for texture data
    glm::vec3* dev_data = nullptr;
    cudaMalloc(&dev_data, texture.width * texture.height * sizeof(glm::vec3));
    cudaMemcpy(dev_data, texture.data, texture.width * texture.height * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    // Copy pointers to device memory
    cudaMemcpy(&(dev_textures->data), &dev_data, sizeof(glm::vec3*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dev_textures->width), &texture.width, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dev_textures->height), &texture.height, sizeof(int), cudaMemcpyHostToDevice);

    checkCUDAError("allocateMemForTexture: copy texture data to device");
}
void allocateMemForMaterial(const Material& material, Material* dev_materials) {

    if (material.diffuseTexture != nullptr) {
        Texture* dev_diffuseTextures = nullptr;
        cudaMalloc(&dev_diffuseTextures, sizeof(Texture));
        allocateMemForTexture(*(material.diffuseTexture), dev_diffuseTextures);
        cudaMemcpy(&(dev_materials->diffuseTexture), &dev_diffuseTextures, sizeof(Texture*), cudaMemcpyHostToDevice);

        checkCUDAError("allocateMemForMaterial: copy diffuse texture");
    }
    if (material.normalTexture != nullptr) {
        Texture* dev_normalTextures = nullptr;
        cudaMalloc(&dev_normalTextures, sizeof(Texture));
        allocateMemForTexture(*(material.normalTexture), dev_normalTextures);
        cudaMemcpy(&(dev_materials->normalTexture), &dev_normalTextures, sizeof(Texture*), cudaMemcpyHostToDevice);
        checkCUDAError("allocateMemForMaterial: copy normal texture");
    }
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    cudaDeviceSynchronize();

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

#if 1
    Mesh* dev_meshes = nullptr;
    cudaMalloc(&dev_meshes, sizeof(Mesh));
	for (int i = 0; i < scene->geoms.size(); i++)
	{
		if (scene->geoms[i].type == MESH)
		{
			Mesh mesh = *scene->geoms[i].mesh;
           
			allocateMemForMesh(mesh, dev_meshes);
			cudaMemcpy(&(dev_geoms[i].mesh), &dev_meshes, sizeof(Mesh*), cudaMemcpyHostToDevice);
			checkCUDAError("pathtraceInit: copy mesh");
		
		}
	}
#endif

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
    Texture* dev_diffuseTextures = nullptr;
    cudaMalloc(&dev_diffuseTextures, sizeof(Texture));
    Texture* dev_normalTextures = nullptr;
    cudaMalloc(&dev_normalTextures, sizeof(Texture));
    for (int i = 0; i < scene->materials.size(); i++)
    {
        Material material = scene->materials[i];
        if (scene->materials[i].diffuseTexture != nullptr) {
            Texture diffuseTexture = *material.diffuseTexture;
            allocateMemForTexture(diffuseTexture, dev_diffuseTextures);
            cudaMemcpy(&(dev_materials[i].diffuseTexture), &dev_diffuseTextures, sizeof(Texture*), cudaMemcpyHostToDevice);
            checkCUDAError("pathtraceInit: copy diffuse texture");
        }
        if (scene->materials[i].normalTexture != nullptr) {
            Texture normalTexture = *material.normalTexture;
            allocateMemForTexture(normalTexture, dev_normalTextures);
            cudaMemcpy(&(dev_materials[i].normalTexture), &dev_normalTextures, sizeof(Texture*), cudaMemcpyHostToDevice);
            checkCUDAError("pathtraceInit: copy normal texture");
        }
    }
    //cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0.0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    //cudaFree(dev_geoms);
    
    // TODO: clean up any extra device memory you created
#if 1
    // free mesh and triangle data
	if (hst_scene != NULL)
	{
		for (int i = 0; i < hst_scene->geoms.size(); i++)
		{
			if (hst_scene->geoms[i].type == MESH)
			{
                Mesh* dev_meshes = nullptr;
                cudaDeviceSynchronize();
				cudaMemcpy(&dev_meshes, &(dev_geoms[i].mesh), sizeof(Mesh*), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
				glm::vec3* dev_vertices = nullptr;
				glm::vec3* dev_normals = nullptr;
				glm::vec3* dev_uvs = nullptr;
				int* dev_indices = nullptr;
				Triangle* dev_triangles = nullptr;
				cudaMemcpy(&dev_vertices, &(dev_meshes->vertices), sizeof(glm::vec3*), cudaMemcpyDeviceToHost);
				cudaFree(dev_vertices);
				cudaMemcpy(&dev_normals, &(dev_meshes->normals), sizeof(glm::vec3*), cudaMemcpyDeviceToHost);
				cudaFree(dev_normals);
				cudaMemcpy(&dev_uvs, &(dev_meshes->uvs), sizeof(glm::vec3*), cudaMemcpyDeviceToHost);
				cudaFree(dev_uvs);
				cudaMemcpy(&dev_indices, &(dev_meshes->indices), sizeof(int*), cudaMemcpyDeviceToHost);
				cudaFree(dev_indices);
				cudaMemcpy(&dev_triangles, &(dev_meshes->triangles), sizeof(Triangle*), cudaMemcpyDeviceToHost);
				cudaFree(dev_triangles);
				cudaFree(dev_meshes);

			}
		}

        // free texture data
        for (int i = 0; i < hst_scene->materials.size(); i++)
        {
            if (hst_scene->materials[i].hasDiffuseTexture)
            {
                Texture* dev_diffuseTextures = nullptr;
                cudaMemcpy(&dev_diffuseTextures, &(dev_materials[i].diffuseTexture), sizeof(Texture*), cudaMemcpyDeviceToHost);
                glm::vec3* dev_data = nullptr;
                cudaMemcpy(&dev_data, &(dev_diffuseTextures->data), sizeof(glm::vec3*), cudaMemcpyDeviceToHost);
                cudaFree(dev_data);
                cudaFree(dev_diffuseTextures);
            }
            if (hst_scene->materials[i].hasNormalTexture)
            {
                Texture* dev_normalTextures = nullptr;
                cudaMemcpy(&dev_normalTextures, &(dev_materials[i].normalTexture), sizeof(Texture*), cudaMemcpyDeviceToHost);
                glm::vec3* dev_data = nullptr;
                cudaMemcpy(&dev_data, &(dev_normalTextures->data), sizeof(glm::vec3*), cudaMemcpyDeviceToHost);
                cudaFree(dev_data);
                cudaFree(dev_normalTextures);
            }
        }
	}
#endif
    cudaDeviceSynchronize();
    cudaFree(dev_geoms);
	checkCUDAError("pathtraceFree: free geoms");
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
		glm::vec3 jitter = glm::vec3(0.0f);
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(-0.5, 0.5);
		jitter = glm::vec3(u01(rng), u01(rng), 0.0f);
        float resolution_x = cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f);
		float resolution_y = cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f);
		jitter = glm::vec3(jitter.x * cam.pixelLength.x, jitter.y * cam.pixelLength.y, 0.0f);
        segment.ray.direction = jitter + glm::normalize(cam.view
            - cam.right * resolution_x
            - cam.up * resolution_y
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec3 uv;
        glm::mat3 TBN;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec3 tmp_uv;
        glm::mat3 tmp_TBN;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?
			else if (geom.type == MESH)
			{
                t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, tmp_TBN, outside);
				//checkCUDAError("computeIntersections: meshIntersectionTest");
			}

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                uv = tmp_uv;
                TBN = tmp_TBN;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].uv = uv;
            //intersections[path_index].TBN = TBN;
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
                //return;
                
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                scatterRay(pathSegments[idx], 
                    getPointOnRay(pathSegments[idx].ray, intersection.t), 
                    intersection.surfaceNormal, 
                    intersection.uv,
                    //intersection.TBN,
                    material, 
                    rng);
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.

        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}

// add the color from each depth to the image
__global__ void accumulateColor(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
        if (iterationPath.remainingBounces <= 0) {
            
            glm::vec3 color = iterationPath.color;
            // gamma correction
            //color = glm::pow(color, glm::vec3(1.0f / 2.2f));
            image[iterationPath.pixelIndex] += color;
        }
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
		glm::vec3 color = iterationPath.color;
        // gamma correction
		//color = glm::pow(color, glm::vec3(1.0f / 2.2f));
        image[iterationPath.pixelIndex] += color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false; // iteration
#if FIREST_BOUNCE
    while (!iterationComplete)
#else
	cudaDeviceSynchronize();
    while (!iterationComplete)
#endif
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

#if STREAM_COMPACTION_INTERSECTION
		// Stream compaction
		// compact after intersection
        // zip operator to compact pathSeg and inteersection simultaneously
		auto first = thrust::make_zip_iterator(thrust::make_tuple(dev_paths, dev_intersections));
		auto last = thrust::make_zip_iterator(thrust::make_tuple(dev_paths + num_paths, dev_intersections + num_paths));
		auto new_end_tuple = thrust::make_zip_iterator(thrust::make_tuple(dev_paths, dev_intersections));

        auto new_end_iter = thrust::copy_if(
            thrust::device,
			first,
			last,
            dev_intersections,
            new_end_tuple,
			hasIntersection()
        );

        cudaDeviceSynchronize();
		num_paths = thrust::get<0>(new_end_iter.get_iterator_tuple()) - dev_paths;
		dev_path_end = dev_paths + num_paths;
#endif

#if SORT_BY_MATERIAL
        // sort pathSeg by material types
		thrust::sort_by_key(
            thrust::device, 
            dev_intersections, 
            dev_intersections + num_paths, 
            dev_paths,
            sortByMaterial());
		cudaDeviceSynchronize();

#endif

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );

        cudaDeviceSynchronize();
#if FIREST_BOUNCE
        iterationComplete = true; // TODO: should be based off stream compaction results.
#else
		accumulateColor << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_image, dev_paths);
        cudaDeviceSynchronize();
#if STREAM_COMPACTION_PATH
        // Stream compaction
		PathSegment* new_end = thrust::remove_if(thrust::device, dev_paths, dev_path_end, hasNoMoreBounces());
        cudaDeviceSynchronize();
		num_paths = new_end - dev_paths;
		dev_path_end = new_end;
        if (num_paths <= 0 || depth > traceDepth) {
            iterationComplete = true;
        }
#endif
        
#endif
        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);
	cudaDeviceSynchronize();

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
