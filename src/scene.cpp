
#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#include "TinyObj/tiny_obj_loader.h"

using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

// write mesh info to Geom
void writeMeshInfo(Geom* geom, std::vector<tinyobj::shape_t>& shapes, std::vector<tinyobj::material_t>& materials) {
    int nextTriangleIndex = 0;
    //Read the information from the vector of shape_ts
    for (unsigned int i = 0; i < shapes.size(); i++)
    {
        std::vector<float>& positions = shapes[i].mesh.positions;
        std::vector<float>& normals = shapes[i].mesh.normals;
        std::vector<float>& uvs = shapes[i].mesh.texcoords;
        std::vector<unsigned int>& indices = shapes[i].mesh.indices;
        for (unsigned int j = 0; j < indices.size(); j += 3)
        {
            glm::vec3 p1(positions[indices[j] * 3], positions[indices[j] * 3 + 1], positions[indices[j] * 3 + 2]);
            glm::vec3 p2(positions[indices[j + 1] * 3], positions[indices[j + 1] * 3 + 1], positions[indices[j + 1] * 3 + 2]);
            glm::vec3 p3(positions[indices[j + 2] * 3], positions[indices[j + 2] * 3 + 1], positions[indices[j + 2] * 3 + 2]);

            Triangle t = Triangle(p1, p2, p3, nextTriangleIndex++);
            //                t.mesh_id = triangle_mesh_id;
            if (normals.size() > 0)
            {
                glm::vec3 n1(normals[indices[j] * 3], normals[indices[j] * 3 + 1], normals[indices[j] * 3 + 2]);
                glm::vec3 n2(normals[indices[j + 1] * 3], normals[indices[j + 1] * 3 + 1], normals[indices[j + 1] * 3 + 2]);
                glm::vec3 n3(normals[indices[j + 2] * 3], normals[indices[j + 2] * 3 + 1], normals[indices[j + 2] * 3 + 2]);
                t.normals[0] = n1;
                t.normals[1] = n2;
                t.normals[2] = n3;
            }
            if (uvs.size() > 0)
            {
                glm::vec3 t1(uvs[indices[j] * 2], uvs[indices[j] * 2 + 1], 0);
                glm::vec3 t2(uvs[indices[j + 1] * 2], uvs[indices[j + 1] * 2 + 1], 0);
                glm::vec3 t3(uvs[indices[j + 2] * 2], uvs[indices[j + 2] * 2 + 1], 0);
                t.uvs[0] = t1;
                t.uvs[1] = t2;
                t.uvs[2] = t3;
            }
            geom->mesh->triangles.push_back(t);
        }
    }
}


void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.hasReflective = 1;
			newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
			float roughness = p["ROUGHNESS"];
			newMaterial.specular.exponent = 1.0 / roughness * roughness;
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
		else if (p["TYPE"] == "Reflective")
		{
			const auto& col = p["RGB"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.hasReflective = 1;
		}
		else if (p["TYPE"] == "Refractive")
		{
			const auto& col = p["RGB"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.hasRefractive = 1;
			newMaterial.indexOfRefraction = p["IOR"];
			printf("IOR: %f\n", newMaterial.indexOfRefraction);
		}
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
		else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
		else if (type == "mesh")
		{
			newGeom.type = MESH;
			std::string meshFilePath = p["FILE"];
            //test mesh file path
			std::cout << meshFilePath << std::endl;
            // load obj file
			newGeom.mesh = new Mesh();
            std::vector<tinyobj::shape_t> shapes; std::vector<tinyobj::material_t> materials;
            std::string errors = tinyobj::LoadObj(shapes, materials, meshFilePath.c_str());
            std::cout << errors << std::endl;
            if (errors.size() == 0) {
				writeMeshInfo(&newGeom, shapes, materials);
			}
            else {
                std::cout << "Error loading mesh: " << errors << std::endl;
            }

		}
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
