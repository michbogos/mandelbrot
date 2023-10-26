#include <raylib.h>
#include <raymath.h>

int main(void)
{
    InitWindow(800, 450, "raylib [core] example - basic window");
    Shader FractalShader = LoadShader(0, "mandelbrot.fs");

    int iresLoc        = GetShaderLocation(FractalShader, "resolution");
    int topLeftloc     = GetShaderLocation(FractalShader, "topLeft");
    int scaleLoc = GetShaderLocation(FractalShader, "scale");

    

    float resolution[2] = {800, 450};

    float topLeft[2] = {0, 0};
    float scale = 1.0f;

    SetShaderValue(FractalShader, iresLoc, resolution, SHADER_UNIFORM_VEC2);

    while (!WindowShouldClose())
    {

        SetShaderValue(FractalShader, topLeftloc, topLeft, SHADER_ATTRIB_VEC2);
        SetShaderValue(FractalShader, scaleLoc, &scale, SHADER_UNIFORM_FLOAT);

        BeginDrawing();
            ClearBackground(BLACK);
            BeginShaderMode(FractalShader);
                    DrawRectangle(0, 0, 800, 450, BLACK);
            EndShaderMode();
            DrawFPS(10, 10);
        EndDrawing();

        if(IsKeyDown(KEY_SPACE)){
            scale *= 1-GetFrameTime();
        }
        if(IsKeyDown(KEY_LEFT)){
            topLeft[0] -= GetFrameTime()*scale;
        }
        if(IsKeyDown(KEY_RIGHT)){
            topLeft[0] += GetFrameTime()*scale;
        }
        if(IsKeyDown(KEY_UP)){
            topLeft[1] += GetFrameTime()*scale;
        }
        if(IsKeyDown(KEY_DOWN)){
            topLeft[1] -= GetFrameTime()*scale;
        }
    }
    UnloadShader(FractalShader);
    CloseWindow();

    return 0;
}