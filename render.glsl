precision mediump float;
uniform float globalTime;
uniform vec2 window;
const float eps = 0.001;
float rand(vec2 n) { 
	return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}
vec3 rgb(float r,float g,float b){
  return vec3(r/255.,g/255.,b/255.);
}
mat4 rotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

vec3 rotate(vec3 v, vec3 axis, float angle) {
	mat4 m = rotationMatrix(axis, angle);
	return (m * vec4(v, 1.0)).xyz;
}
//	Classic Perlin 2D Noise 
//	by Stefan Gustavson
//
vec2 fade(vec2 t) {return t*t*t*(t*(t*6.0-15.0)+10.0);}
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
float cnoise(vec2 P){
    vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
    vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
    Pi = mod(Pi, 289.0); // To avoid truncation effects in permutation
    vec4 ix = Pi.xzxz;
    vec4 iy = Pi.yyww;
    vec4 fx = Pf.xzxz;
    vec4 fy = Pf.yyww;
    vec4 i = permute(permute(ix) + iy);
    vec4 gx = 2.0 * fract(i * 0.0243902439) - 1.0; // 1/41 = 0.024...
    vec4 gy = abs(gx) - 0.5;
    vec4 tx = floor(gx + 0.5);
    gx = gx - tx;
    vec2 g00 = vec2(gx.x,gy.x);
    vec2 g10 = vec2(gx.y,gy.y);
    vec2 g01 = vec2(gx.z,gy.z);
    vec2 g11 = vec2(gx.w,gy.w);
    vec4 norm = 1.79284291400159 - 0.85373472095314 * 
    vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11));
    g00 *= norm.x;
    g01 *= norm.y;
    g10 *= norm.z;
    g11 *= norm.w;
    float n00 = dot(g00, vec2(fx.x, fy.x));
    float n10 = dot(g10, vec2(fx.y, fy.y));
    float n01 = dot(g01, vec2(fx.z, fy.z));
    float n11 = dot(g11, vec2(fx.w, fy.w));
    vec2 fade_xy = fade(Pf.xy);
    vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
    float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
    return 2.3 * n_xy;
}
float minh = 0.0, maxh = 6.0;
vec3 nn = vec3(0);

float hash(float n)
{
    return fract(sin(n) * 43758.5453);
}

float noise(vec3 p)
{
    return hash(p.x + p.y*57.0 + p.z*117.0);
}

float valnoise(vec3 p)
{
    vec3 c = floor(p);
    vec3 f = smoothstep(0., 1., fract(p));
    return mix(
        mix (mix(noise(c + vec3(0, 0, 0)), noise(c + vec3(1, 0, 0)), f.x),
             mix(noise(c + vec3(0, 1, 0)), noise(c + vec3(1, 1, 0)), f.x), f.y),
        mix (mix(noise(c + vec3(0, 0, 1)), noise(c + vec3(1, 0, 1)), f.x),
             mix(noise(c + vec3(0, 1, 1)), noise(c + vec3(1, 1, 1)), f.x), f.y),
        f.z);
}

float fbm(vec3 p)
{
    float f = 0.;
    for(int i = 0; i < 5; ++i)
        f += (valnoise(p * exp2(float(i))) - .5) / exp2(float(i));
    return f;
}

float height2(vec2 p)
{
    float h = mix(minh, maxh * 1.3, pow(clamp(.2 + .8 * fbm(vec3(p / 6., 0.)), 0., 1.), 1.3));
    h += valnoise(vec3(p, .3));
    return h;
}
float sdf(vec3 v){
    return min(length(v - vec3(-0.3,0,0) + vec3(-sin(globalTime),-cos(globalTime),0))  - 0.6, length( v - vec3(0.6,0,1)) - 0.6  ) ;
}
float unionsdf(float a, float b){
    return min(a,b);
}
float intersectsdf(float a, float b){
    return max(a,b);
}
float sea(vec3 v){
    float x = v.x * 2.0+ globalTime;
    float y = v.z * 2.0+ globalTime;
    return v.y - (sin(x) + cos(y))*.07 + 0.5;
}
float terrain(vec3 v){
    const float scale = 1.0;
    float x = v.x / scale;
    float z = v.z / scale;
    //return  (cnoise(vec2(x,z)) + 1.0) * 3.0;
     return height2(vec2(x,z));
}
float heightmap(vec3 v){
    return v.y - terrain(v);
}
vec3 estimatenorm(vec3 v){
    const float eps = 0.0001;
    return normalize(vec3(
        sdf(vec3(v.x + eps,v.y,v.z)) - sdf(vec3(v.x - eps,v.y,v.z)),
        sdf(vec3(v.x, v.y + eps,v.z)) - sdf(vec3(v.x, v.y - eps,v.z)),
        sdf(vec3(v.x, v.y,v.z + eps)) - sdf(vec3(v.x, v.y,v.z - eps))));
}
float cloud(vec2 v){
    const float scale = 5.0;
    v /= scale;
    return clamp((cnoise(v)),0.0,1.);
}
vec3 tracecloud(vec3 p ,vec3 dir){
    const float cloudheight = 4.0;
    float t = (p.y - cloudheight) / dir.y;
    p += t * dir;
    return vec3(1.,1.,1.) * cloud(vec2(p.x,p.z));
}
bool shadow(vec3 p, vec3 dir){
    for(float i = 0.0;i<20.0;i+=0.1){
        if(heightmap(p)<0.0)
            return false;
        p += 0.5 * dir;
    }
    return true;
}
vec3 htrace(vec3 p, vec3 dir){
    bool hit = false;
    vec3 o = p;
    float t = .0;
    float t0 = t;
    for(float i=0.0;i<40.0;i+=1.0){
        if(hit)break;
        float h = heightmap(p);
        if(h <0.0){
          hit = true;
          break;
        }
        t0 = t;
        t += 4.0 + t/100.0;
        p = o + t * dir;
        
    }
    if(!hit)
    return vec3(0,0,0);
    float distmax = t;
    float distmin = t0;
    for(float j = 0.;j<16.;j+=1.0){
        float distmid = 0.5 * (distmax + distmin);
        p = o + distmid * dir;
        float h = heightmap(p);
        if(abs(h)<eps){
            break;
        }
        if(h > .0){
            distmin = distmid;
        }else{
            distmax = distmid;
        }
    }
    if(heightmap(p) > 1.0){
      return vec3(0.,0.,.0);
    }
    vec3 normal;
    vec3 dx = vec3(eps,.0,.0),dz = vec3(.0,.0,eps);
    normal.x = heightmap(p + dx) - heightmap(p - dx);
    normal.z = heightmap(p + dz) -  heightmap(p - dz);
    normal.x /= 2.0*eps;
    normal.z /= 2.0*eps;
    normal.y = 1.0;
    normal = normalize(normal);
    vec3 d = normalize(vec3(0.4,1.0,0.1));  
    float height = p.y +0.25;
    vec3 color = vec3(1,1,1);
    color *= mix(vec3(1, .8, .5) / 2., vec3(.3, 1, .3) / 4., 1. - clamp(height / 2., 0., 1.));
    color = mix(color, vec3(1) * .7, pow(clamp((height - 2.5) / 2., 0., 1.), 2.));
    if(!shadow(p, -d))
        color *= max(0.0,1.0 * dot(d,normal));
    color += tracecloud(o, dir);
    return color;
}
vec3 sky(vec3 p, vec3 dir){
    if(dir.y <0.){
        return vec3(0.0,0.0,.0);
    }
    return vec3(.25,.41,.64) *1.3;
}

vec3 trace(vec3 p, vec3 dir){
  vec3 light = vec3(sin(globalTime),1,cos(globalTime));
    for(float i = 0.0; i<10.0; i+=1.0){
        float dist = sdf(p);
        if(dist < 0.1){
            vec3 n = estimatenorm(p);
            vec3 d = normalize(light - p);  
            vec3 color = rgb(0.,44.,133.);
            return color * max(0.0, dot(d,n));
        }
        p += dir *dist;        
    }
    return vec3(0.0,0.0,0.0);
}
vec3 raymarch(float x,float y){
    float z = -1.0+globalTime*.5;
    float height = 10.0;//terrain(vec3(0,0,z));
    vec3 p = vec3(0,height,z);
    vec3 dir = normalize(vec3(x,y,-1.0)-vec3(0.0,0.0,-3));
    dir = rotate(dir, vec3(1,0,0), -3.1415 / 4.0);
    dir = rotate(dir,vec3(0,1,0), globalTime * 0.1);
    vec3 color;
    color = htrace(p,dir);
   // color += sky(p,dir);
    return color;
   
}

void main() {
    float x = gl_FragCoord.x/window.y - 0.3;
    float y = gl_FragCoord.y/window.y;
    x = 2.0*x - 1.0;
    y = 2.0*y - 1.0;
    gl_FragColor = vec4(raymarch(x,y),1.0);
}